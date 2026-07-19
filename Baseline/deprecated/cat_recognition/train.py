"""YOLOv8n 猫识别模型训练脚本。

在 NVIDIA RTX A2000 Laptop GPU (4GB VRAM) 上训练轻量级 YOLOv8n 模型。

核心优化：
- batch_size=32，充分利用 4GB VRAM
- FP16 混合精度训练（RTX A2000 原生支持）
- 早停 + 模型检查点
- 针对教室场景的数据增强
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练猫识别 YOLOv8n 模型")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--data", default="./datasets/cat_yolo/dataset.yaml", help="数据集 YAML 路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（覆盖配置文件）")
    parser.add_argument("--batch", type=int, default=None, help="批次大小（覆盖配置文件）")
    parser.add_argument("--resume", action="store_true", help="从上次检查点恢复训练")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]

    # 训练参数
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch or train_cfg["batch_size"]

    # 加载预训练模型
    model_name = model_cfg["name"]  # "yolov8n"
    print(f"加载预训练模型: {model_name}.pt")
    model = YOLO(f"{model_name}.pt")

    # 训练
    print(f"\n=== 开始训练 ===")
    print(f"数据集: {args.data}")
    print(f"设备: {train_cfg['device']} (NVIDIA RTX A2000 Laptop GPU)")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"图像尺寸: {train_cfg['imgsz']}×{train_cfg['imgsz']}")

    results = model.train(
        # 数据
        data=args.data,

        # 训练参数
        epochs=epochs,
        batch=batch_size,
        imgsz=train_cfg["imgsz"],
        device=train_cfg["device"],

        # 优化器
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        warmup_momentum=train_cfg["warmup_momentum"],
        warmup_bias_lr=train_cfg["warmup_bias_lr"],

        # 混合精度（RTX A2000 原生支持 FP16）
        amp=True,

        # 数据增强
        hsv_h=train_cfg["augmentation"]["hsv_h"],
        hsv_s=train_cfg["augmentation"]["hsv_s"],
        hsv_v=train_cfg["augmentation"]["hsv_v"],
        degrees=train_cfg["augmentation"]["degrees"],
        translate=train_cfg["augmentation"]["translate"],
        scale=train_cfg["augmentation"]["scale"],
        shear=train_cfg["augmentation"]["shear"],
        perspective=train_cfg["augmentation"]["perspective"],
        flipud=train_cfg["augmentation"]["flipud"],
        fliplr=train_cfg["augmentation"]["fliplr"],
        mosaic=train_cfg["augmentation"]["mosaic"],
        mixup=train_cfg["augmentation"]["mixup"],

        # 早停与保存
        patience=train_cfg["patience"],
        save_period=train_cfg["save_period"],
        save=True,

        # 日志
        project="runs/cat_detection",
        name="yolov8n_cat",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n训练完成！")
    print(f"最佳模型: runs/cat_detection/yolov8n_cat/weights/best.pt")

    # 验证
    print(f"\n=== 验证模型 ===")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
