"""Train YOLOv11n-P2 on pens, erasers, and building blocks.

Validates source annotations, creates a deterministic Ultralytics dataset,
then transfers COCO weights to a custom P2 small-object detection model.
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml

CLASS_NAMES = {0: "pen", 1: "eraser", 2: "building_block"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
logger = logging.getLogger(__name__)


def configure_logging(log_file: Path) -> None:
    """Write preparation and training progress to terminal and a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )


def find_label(image_path: Path, data_root: Path) -> Path | None:
    """Find an adjacent label or its labels/ mirror in standard YOLO folders."""
    adjacent = image_path.with_suffix(".txt")
    if adjacent.exists():
        return adjacent
    parts = list(image_path.relative_to(data_root).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        mirrored = data_root.joinpath(*parts).with_suffix(".txt")
        if mirrored.exists():
            return mirrored
    return None


def label_error(label_path: Path) -> str | None:
    """Validate one YOLO label file and return its first error, if any."""
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return f"cannot read UTF-8 label: {error}"
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # Empty label files represent valid background images.
        fields = line.split()
        if len(fields) != 5:
            return f"line {line_number}: expected 5 fields, got {len(fields)}"
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            return f"line {line_number}: values must be numeric"
        if class_id not in CLASS_NAMES:
            return f"line {line_number}: class {class_id} is outside 0..2"
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            return f"line {line_number}: normalized boxes must fit in [0, 1]"
    return None


def discover_samples(data_root: Path, allow_unlabeled: bool) -> list[tuple[Path, Path | None]]:
    """Find valid image-label pairs, rejecting missing or malformed labels."""
    images = sorted(path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {data_root}")
    samples: list[tuple[Path, Path | None]] = []
    missing: list[Path] = []
    broken: list[str] = []
    for image in images:
        label = find_label(image, data_root)
        if label is None:
            missing.append(image)
            if allow_unlabeled:
                samples.append((image, None))
            continue
        error = label_error(label)
        if error:
            broken.append(f"{label}: {error}")
        else:
            samples.append((image, label))
    if missing and not allow_unlabeled:
        preview = "\n  ".join(str(path) for path in missing[:10])
        raise ValueError(f"{len(missing)} images lack a .txt label:\n  {preview}\nUse --allow-unlabeled for intentional background images.")
    if broken:
        raise ValueError(f"{len(broken)} malformed labels:\n  " + "\n  ".join(broken[:10]))
    if len(samples) < 2:
        raise ValueError("At least two valid images are required for a train/validation split.")
    logger.info("Data validation passed: %d usable images, %d without labels", len(samples), len(missing))
    return samples


def copy_split(samples: Iterable[tuple[Path, Path | None]], data_root: Path, output_root: Path, split: str) -> int:
    """Materialize images/<split> and labels/<split> with identical subpaths."""
    count = 0
    for image, label in samples:
        relative = image.relative_to(data_root)
        output_image = output_root / "images" / split / relative
        output_label = output_root / "labels" / split / relative.with_suffix(".txt")
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output_image)
        if label is None:
            output_label.write_text("", encoding="utf-8")
        else:
            shutil.copy2(label, output_label)
        count += 1
    return count


def prepare_dataset(data_root: Path, output_root: Path, val_ratio: float, seed: int, allow_unlabeled: bool) -> Path:
    """Validate and deterministically create the required 85/15 YOLO dataset."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be strictly between 0 and 1")
    samples = discover_samples(data_root, allow_unlabeled)
    random.Random(seed).shuffle(samples)
    validation_size = min(max(1, round(len(samples) * val_ratio)), len(samples) - 1)
    validation, training = samples[:validation_size], samples[validation_size:]
    if output_root.exists():
        shutil.rmtree(output_root)
    train_count = copy_split(training, data_root, output_root, "train")
    val_count = copy_split(validation, data_root, output_root, "val")
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(output_root.resolve()), "train": "images/train", "val": "images/val",
        "nc": 3, "names": CLASS_NAMES,
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("Prepared split with seed %d: train=%d, val=%d", seed, train_count, val_count)
    logger.info("Generated dataset configuration: %s", dataset_yaml)
    return dataset_yaml


def run_training(args: argparse.Namespace, dataset_yaml: Path) -> None:
    """Initialize custom P2 model, partially load COCO weights, and train it."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install Ultralytics and CUDA-enabled PyTorch before training.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was not detected; this RTX 4070 training configuration requires CUDA.")
    model_yaml = args.model_yaml.resolve()
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {model_yaml}")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Building P2 architecture: %s", model_yaml)
    model = YOLO(str(model_yaml))
    logger.info("Loading COCO pretraining checkpoint: %s", args.weights)
    model.load(args.weights)
    results = model.train(
        data=str(dataset_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=0, workers=args.workers, project=args.project, name=args.name, exist_ok=args.exist_ok,
        pretrained=False, amp=True, patience=args.patience, cos_lr=True, lr0=args.lr0, lrf=args.lrf,
        weight_decay=args.weight_decay, mosaic=1.0, scale=args.scale, copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic, seed=args.seed, deterministic=True, plots=True, verbose=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training finished. Best checkpoint: %s", best_weights)
    if best_weights.exists():
        metrics = YOLO(str(best_weights)).val(data=str(dataset_yaml), imgsz=args.imgsz, device=0, split="val")
        logger.info("Best validation metrics: mAP50=%.4f, mAP50-95=%.4f", metrics.box.map50, metrics.box.map)
    else:
        logger.warning("best.pt was not produced. Review the run at: %s", results.save_dir)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11n-P2 for small hand-held object detection.")
    parser.add_argument("--data-path", type=Path, default=root / "collected_images")
    parser.add_argument("--dataset-dir", type=Path, default=root / "prepared_yolo_dataset")
    parser.add_argument("--model-yaml", type=Path, default=root / "yolov11n-p2.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="COCO-pretrained YOLO11n checkpoint.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16, help="16 suits 12 GB VRAM; use -1 for auto-batch.")
    parser.add_argument("--imgsz", type=int, default=800, choices=[640, 800, 1024])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--copy-paste", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--allow-unlabeled", action="store_true")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolov11n_p2_small_objects")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_path = args.data_path.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    configure_logging(args.dataset_dir.parent / "train_yolo_v11_p2.log")
    try:
        dataset_yaml = prepare_dataset(args.data_path, args.dataset_dir, args.val_ratio, args.seed, args.allow_unlabeled)
        run_training(args, dataset_yaml)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("Training aborted: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()