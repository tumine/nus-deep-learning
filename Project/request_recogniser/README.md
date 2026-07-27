# Request Recogniser Training

This directory trains a YOLO object detector from the images and YOLO-format bounding-box labels in `request_dataset`.

## Dataset

The existing dataset is already arranged in the format used by Ultralytics YOLO:

```text
request_dataset/
  data.yaml
  train/images/ and train/labels/
  valid/images/ and valid/labels/
  test/images/ and test/labels/
```

`data.yaml` defines these classes:

| Class ID | Class name |
| --- | --- |
| 0 | `block` |
| 1 | `pencil` |
| 2 | `eraser` |

Each label file must share the image filename and contain one normalized YOLO bounding box per line:

```text
class_id x_center y_center width height
```

## Install dependencies

From the repository root, create or activate a Python 3.11+ environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

For NVIDIA GPU training, install the PyTorch build that matches the local CUDA driver before `pip install -e .`. Check that GPU support is available:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Train

Run the command from the repository root. By default, `train.py` uses the included `request_dataset/yolo26n.pt` as the starting checkpoint and saves output under `request_recogniser/runs/request_detector/`.

```powershell
python Project/request_recogniser/train.py
```

Useful options:

```powershell
# Lower memory use or force CPU execution
python Project/request_recogniser/train.py --batch 8 --imgsz 640 --device cpu

# Continue an interrupted run with the same output name
python Project/request_recogniser/train.py --resume

# Use another compatible pre-trained checkpoint
python Project/request_recogniser/train.py --weights path\to\yolo_model.pt --epochs 150
```

Before starting YOLO, the script checks that the training and validation image directories exist and that each contains the same number of image and label files. Training figures, metrics, and `weights/best.pt` are stored in:

```text
Project/request_recogniser/runs/request_detector/
```

## Evaluate and predict

Evaluate the best checkpoint on the held-out test split:

```powershell
python Project/request_recogniser/test.py
```

Evaluate validation data instead, or select a particular checkpoint:

```powershell
python Project/request_recogniser/test.py --split val
python Project/request_recogniser/test.py --weights Project/request_recogniser/runs/request_detector/weights/best.pt
```

To both report test metrics and save annotated predictions for a separate image, directory, or video:

```powershell
python Project/request_recogniser/test.py --source path\to\image_or_video.jpg --conf 0.25
```

Prediction output is written below `Project/request_recogniser/runs/request_detector_predictions/`.