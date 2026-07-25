"""Prepare and optionally run a zero-shot YOLO-World detector.

This workflow uses no local bounding-box annotations and performs no fine-tuning.
It embeds text prompts in a pretrained open-vocabulary model for deployment.
"""

import argparse
import logging
import time
import urllib.error
import sys
from pathlib import Path

PROMPTS = ["pen", "eraser", "building block"]
logger = logging.getLogger(__name__)


def configure_logging(output_dir: Path) -> None:
    """Configure terminal and file logging for model preparation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "prepare_yolo_world.log", encoding="utf-8"),
        ],
    )


def load_prompted_model(weights: str, prompts: list[str], output_path: Path, download_retries: int):
    """Load an open-vocabulary checkpoint, bind prompts, and save it locally."""
    try:
        import torch
        from ultralytics import YOLOWorld
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics with YOLOWorld support and CUDA-enabled PyTorch are required. "
            "Install project dependencies before preparing the model."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was not detected. YOLO-World preparation requires the RTX 4070 CUDA environment.")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Loading open-vocabulary checkpoint: %s", weights)
    for attempt in range(1, download_retries + 1):
        try:
            model = YOLOWorld(weights)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == download_retries:
                raise RuntimeError(
                    f"Could not download checkpoint '{weights}' after {download_retries} attempts. "
                    "The HTTPS connection was interrupted. Check the proxy/firewall connection to GitHub, "
                    "or download the checkpoint manually and pass its local path with --weights."
                ) from error
            delay_seconds = 2 ** (attempt - 1)
            logger.warning(
                "Checkpoint download failed (%s/%s): %s. Retrying in %s seconds.",
                attempt, download_retries, error, delay_seconds,
            )
            time.sleep(delay_seconds)
    model.set_classes(prompts)
    model.save(str(output_path))
    logger.info("Saved model with embedded prompts %s: %s", prompts, output_path)
    return model


def run_prediction(model, source: Path, output_dir: Path, imgsz: int, confidence: float) -> None:
    """Run zero-shot prediction on an image, directory, video, or camera source."""
    if not source.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {source}")
    logger.info("Starting zero-shot detection; labels are not required: %s", source)
    model.predict(
        source=str(source), imgsz=imgsz, conf=confidence, device=0,
        save=True, save_txt=True, project=str(output_dir), name="predictions",
        exist_ok=True, verbose=True,
    )
    logger.info("Annotated images and prediction text files: %s", output_dir / "predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a zero-shot YOLO-World detector for pen, eraser, and building block prompts."
    )
    parser.add_argument("--weights", default="yolov8s-worldv2.pt", help="Pretrained YOLO-World checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "zero_shot_output")
    parser.add_argument("--model-name", default="yolov8s_world_school_objects.pt")
    parser.add_argument("--source", type=Path, default=None, help="Optional unannotated image, directory, or video for inference.")
    parser.add_argument("--imgsz", type=int, default=800, choices=[640, 800, 1024])
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold for zero-shot inference.")
    parser.add_argument(
        "--download-retries", type=int, default=3,
        help="Number of attempts to download a missing checkpoint (default: 3).",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=PROMPTS,
        help="English object prompts. The prompt order defines output class IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.conf < 1:
        raise SystemExit("--conf must be strictly between 0 and 1.")
    if args.download_retries < 1:
        raise SystemExit("--download-retries must be at least 1.")
    if not args.prompts:
        raise SystemExit("At least one text prompt is required.")
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    try:
        model = load_prompted_model(
            args.weights, args.prompts, output_dir / args.model_name, args.download_retries
        )
        if args.source is not None:
            run_prediction(model, args.source.resolve(), output_dir, args.imgsz, args.conf)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("Zero-shot preparation aborted: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()