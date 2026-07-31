"""Evaluate a trained request detector and optionally save predictions."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "request_dataset"
DEFAULT_WEIGHTS = ROOT / "runs" / "request_detector" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate the request object detector.")
	parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument("--data", type=Path, default=DATASET_DIR / "data.yaml")
	parser.add_argument("--split", choices=("val", "test"), default="test")
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", default=None, help="GPU ID such as 0, or cpu.")
	parser.add_argument("--source", type=Path, help="Optional image, image directory, or video for prediction output.")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--project", type=Path, default=ROOT / "runs")
	parser.add_argument("--name", default="request_detector_predictions")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	weights_path = args.weights.resolve()
	data_path = args.data.resolve()
	if not weights_path.is_file():
		raise FileNotFoundError(f"Trained weights do not exist: {weights_path}")
	if not data_path.is_file():
		raise FileNotFoundError(f"Dataset configuration does not exist: {data_path}")

	from ultralytics import YOLO

	model = YOLO(str(weights_path))
	validation_options = {"data": str(data_path), "split": args.split, "imgsz": args.imgsz}
	if args.device is not None:
		validation_options["device"] = args.device
	metrics = model.val(**validation_options)
	print(f"{args.split} mAP50: {metrics.box.map50:.4f}")
	print(f"{args.split} mAP50-95: {metrics.box.map:.4f}")

	if args.source is not None:
		prediction_options = {
			"source": str(args.source.resolve()),
			"imgsz": args.imgsz,
			"conf": args.conf,
			"save": True,
			"project": str(args.project.resolve()),
			"name": args.name,
		}
		if args.device is not None:
			prediction_options["device"] = args.device
		results = model.predict(**prediction_options)
		print(f"Saved {len(results)} predictions to: {results[0].save_dir if results else args.project / args.name}")


if __name__ == "__main__":
	main()
