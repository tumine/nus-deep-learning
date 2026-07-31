"""Train a YOLO object detector with the local request_dataset annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "request_dataset"
DEFAULT_WEIGHTS = DATASET_DIR / "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    """Parse training options while keeping paths independent of the shell CWD."""
    parser = argparse.ArgumentParser(description="Train the request object detector.")
    parser.add_argument("--data", type=Path, default=DATASET_DIR / "data.yaml")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="GPU ID such as 0, or cpu. Defaults to Ultralytics auto-selection.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="request_detector")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true", help="Resume the run named by --project and --name.")
    return parser.parse_args()


def validate_dataset(data_path: Path) -> dict:
    """Check that the configured dataset splits exist before downloading or training."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {data_path}")

    with data_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not config.get("names"):
        raise ValueError(f"{data_path} must define class names in 'names'.")

    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    for split in ("train", "val"):
        relative_images = config.get(split)
        if not relative_images:
            raise ValueError(f"{data_path} is missing the '{split}' image path.")
        image_dir = dataset_root / relative_images
        label_dir = image_dir.parent / "labels"
        image_count = sum(path.is_file() for path in image_dir.glob("*"))
        label_count = sum(path.is_file() for path in label_dir.glob("*.txt"))
        if not image_dir.is_dir() or image_count == 0:
            raise FileNotFoundError(f"No images found for '{split}': {image_dir}")
        if label_count != image_count:
            raise ValueError(f"{split} has {image_count} images but {label_count} label files.")
        print(f"{split}: {image_count} images, {label_count} labels")
    return config


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    weights_path = args.weights.resolve()
    validate_dataset(data_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Initial weights do not exist: {weights_path}\n"
            "Download a compatible YOLO checkpoint or pass it with --weights."
        )

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    train_options = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "patience": args.patience,
        "resume": args.resume,
        "exist_ok": args.resume,
        "plots": True,
    }
    if args.device is not None:
        train_options["device"] = args.device

    results = model.train(**train_options)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
