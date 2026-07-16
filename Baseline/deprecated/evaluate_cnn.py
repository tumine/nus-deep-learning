"""
猫品种识别模型评估脚本
======================

训练完成后，用此脚本对模型进行正式的测试评估。

两类评估场景：

场景 A：训练时划分了独立测试集（--test-ratio > 0）
   → 加载 hold-out 测试集进行评估，结果完全无偏
   
场景 B：训练时仅划分了训练/验证集（默认）
   → 在验证集上评估，并给出相应提醒

评估指标：
  - 总体准确率（Accuracy）
  - 各类别精确率、召回率、F1 分数
  - 混淆矩阵（Confusion Matrix）
  - Top-K 准确率（Top-1, Top-3）
  - 推理速度（单张延迟 + 吞吐量）

用法：
    # 在测试集上评估（训练时指定了 --test-ratio 0.1）
    python evaluate_cnn.py --model outputs/resnet50_cat_xxx/checkpoints/best_model.pth

    # 指定测试集图像清单（训练自动生成 test_image_paths.txt）
    python evaluate_cnn.py --model best_model.pth --test-list outputs/.../test_image_paths.txt

    # 单张图片推理
    python evaluate_cnn.py --model best_model.pth --image cat.jpg

    # 不启用 FP16 加速
    python evaluate_cnn.py --model best_model.pth --no-half
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# 品种信息
# ============================================================

BREED_CN = {
    "ragdoll": "布偶猫",
    "singapura": "新加坡猫",
    "persian": "波斯猫",
    "sphynx": "斯芬克斯猫",
    "pallas": "兔狲",
}


# ============================================================
# GPU 检测
# ============================================================

def get_device(use_half: bool = True) -> tuple[torch.device, bool]:
    """获取计算设备，优先使用 CUDA GPU。"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {name} ({vram:.1f} GB VRAM)")
        logger.info(f"FP16 推理: {'✅ 启用' if use_half else '❌ 禁用'}")
        return device, use_half
    else:
        logger.warning("CUDA 不可用，使用 CPU 推理")
        return torch.device("cpu"), False


# ============================================================
# 模型加载
# ============================================================

def load_model(checkpoint_path: str, num_classes: int = 5) -> tuple[nn.Module, int, list[str]]:
    """加载训练好的模型（支持 ResNet-50 / ConvNeXt / EfficientNetV2）。

    Args:
        checkpoint_path: 检查点文件路径 (.pth)
        num_classes: 类别数（默认 5，从 checkpoint 读取）

    Returns:
        (model, input_size, class_names)
    """
    logger.info(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "resnet50")
    input_size = checkpoint.get("input_size", 224)
    class_info = checkpoint.get("class_info", {})
    class_names = class_info.get("classes", ["pallas", "persian", "ragdoll", "singapura", "sphynx"])
    actual_num_classes = len(class_names)

    logger.info(f"  backbone: {backbone}, input_size: {input_size}")
    logger.info(f"  类别: {class_names}")

    # 根据 backbone 构建模型
    if backbone.startswith("convnext"):
        dropout_p = 0.6 if "base" in backbone else 0.5
        pretrained_weights = getattr(
            models, f"ConvNeXt_{backbone.split('_')[1].capitalize()}_Weights"
        ).IMAGENET1K_V1
        model = getattr(models, f"convnext_{backbone.split('_')[1]}")(weights=pretrained_weights)
        classifier_in = model.classifier[2].in_features
        head_attr = "classifier"
    elif backbone.startswith("efficientnet_v2"):
        dropout_p = 0.6 if backbone.endswith("_m") else 0.5
        pretrained_weights = getattr(
            models, f"EfficientNet_V2_{backbone.split('_')[2].upper()}_Weights"
        ).IMAGENET1K_V1
        model = getattr(models, f"efficientnet_v2_{backbone.split('_')[2]}")(weights=pretrained_weights)
        classifier_in = model.classifier[1].in_features
        head_attr = "classifier"
    else:  # resnet50
        dropout_p = 0.6
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        classifier_in = model.fc.in_features
        head_attr = "fc"

    # 构建与训练时相同的分类头
    new_head = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(classifier_in, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p * 0.8),
        nn.Linear(512, actual_num_classes),
    )
    setattr(model, head_attr, new_head)

    # 加载权重
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  参数量: {total_params:,}")
    logger.info(f"  模型加载完成 ✅")

    return model, input_size, class_names


# ============================================================
# 测试集构建
# ============================================================

class TestImageDataset(Dataset):
    """从路径清单文件构建测试集。

    路径清单格式（由 train_cnn.py 自动生成）：
        /path/to/image.jpg\tlabel_index\tclass_name
    """

    def __init__(self, list_file: str, input_size: int = 224):
        self.samples: list[tuple[str, int, str]] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    class_name = parts[2] if len(parts) >= 3 else ""
                    if Path(img_path).exists():
                        self.samples.append((img_path, label, class_name))

        logger.info(f"测试集清单: {list_file} ({len(self.samples)} 张)")

        # 与训练验证时一致的预处理
        self.transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label, img_path


# ============================================================
# 核心评估函数
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_half: bool,
    class_names: list[str],
) -> dict:
    """在测试集上全面评估模型。

    Returns:
        包含所有评估指标的字典
    """
    model = model.to(device)
    if use_half:
        model = model.half()

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []  # 每个样本的类别概率分布
    latencies: list[float] = []
    correct_per_class: dict[str, int] = {name: 0 for name in class_names}
    total_per_class: dict[str, int] = {name: 0 for name in class_names}

    logger.info(f"\n正在评估 {len(dataloader.dataset)} 张图片...")

    start_time = time.time()

    for images, labels, paths in dataloader:
        images = images.to(device, non_blocking=True)
        if use_half:
            images = images.half()

        # 单次推理计时
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000 / len(images))  # ms per image

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs)

        # 打印进度
        if len(all_preds) % 100 == 0:
            current_acc = sum(
                1 for p, l in zip(all_preds, all_labels) if p == l
            ) / len(all_preds)
            logger.info(f"  已评估: {len(all_preds)}/{len(dataloader.dataset)} "
                        f"(当前准确率: {current_acc*100:.2f}%)")

    total_time = time.time() - start_time
    logger.info(f"评估完成，总耗时: {total_time:.1f}s")

    # ================================================================
    # 计算指标
    # ================================================================
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # 总体准确率
    overall_acc = float((all_preds_np == all_labels_np).mean())

    # 各类别统计
    num_classes = len(class_names)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    class_accs: dict[str, float] = {}

    for c in range(num_classes):
        # True Positive: 预测为 c 且标签为 c
        tp = int(((all_preds_np == c) & (all_labels_np == c)).sum())
        # False Positive: 预测为 c 但标签不是 c
        fp = int(((all_preds_np == c) & (all_labels_np != c)).sum())
        # False Negative: 预测不是 c 但标签是 c
        fn = int(((all_preds_np != c) & (all_labels_np == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # 单类准确率
        class_mask = all_labels_np == c
        class_acc = float((all_preds_np[class_mask] == c).mean()) if class_mask.sum() > 0 else 0.0
        class_accs[class_names[c]] = class_acc

        total_per_class[class_names[c]] = int(class_mask.sum())
        correct_per_class[class_names[c]] = int((all_preds_np[class_mask] == c).sum())

    # 宏平均
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Top-3 准确率
    all_probs_np = np.array(all_probs)
    top3_preds = np.argsort(all_probs_np, axis=1)[:, -3:]
    top3_correct = sum(1 for i, l in enumerate(all_labels) if l in top3_preds[i])
    top3_acc = top3_correct / len(all_labels)

    # 延迟统计
    latencies_arr = np.array(latencies)
    latency_stats = {
        "mean_ms": float(np.mean(latencies_arr)),
        "median_ms": float(np.median(latencies_arr)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "min_ms": float(np.min(latencies_arr)),
        "max_ms": float(np.max(latencies_arr)),
        "fps": float(1000.0 / np.mean(latencies_arr)),
    }

    # ================================================================
    # 构建结果
    # ================================================================
    results = {
        "dataset_size": len(all_labels),
        "num_classes": num_classes,
        "class_names": class_names,
        # 总体指标
        "overall_accuracy": overall_acc,
        "top3_accuracy": top3_acc,
        # 宏平均
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        # 各类别详情
        "per_class": {
            name: {
                "precision": precisions[i],
                "recall": recalls[i],
                "f1": f1s[i],
                "accuracy": class_accs[name],
                "total": total_per_class[name],
                "correct": correct_per_class[name],
            }
            for i, name in enumerate(class_names)
        },
        # 延迟
        "latency": latency_stats,
        # 原始数据（供外部使用）
        "_predictions": all_preds,
        "_labels": all_labels,
    }

    return results


def build_confusion_matrix(predictions: list[int], labels: list[int],
                           num_classes: int) -> np.ndarray:
    """构建混淆矩阵（类别数 × 类别数）。"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(predictions, labels):
        cm[l][p] += 1
    return cm


# ============================================================
# 结果输出
# ============================================================

def print_results(results: dict):
    """格式化输出评估结果。"""
    class_names = results["class_names"]

    print("\n" + "=" * 70)
    print("                    模型评估报告")
    print("=" * 70)

    # 总体指标
    print(f"\n📊 总体指标（{results['dataset_size']} 张测试图片）")
    print(f"  {'─' * 40}")
    print(f"  总体准确率 (Overall Accuracy): {results['overall_accuracy']*100:.2f}%")
    print(f"  Top-3 准确率 (Top-3 Accuracy): {results['top3_accuracy']*100:.2f}%")
    print(f"  宏平均精确率 (Macro Precision): {results['macro_precision']*100:.2f}%")
    print(f"  宏平均召回率 (Macro Recall):    {results['macro_recall']*100:.2f}%")
    print(f"  宏平均 F1 (Macro F1):          {results['macro_f1']:.4f}")

    # 各类别详情
    print(f"\n📋 各类别详情")
    print(f"  {'类别':<14} {'精确率':>8} {'召回率':>8} {'F1':>8} {'准确率':>8} {'样本数':>6}")
    print(f"  {'─' * 60}")
    for name in class_names:
        pc = results["per_class"][name]
        cn_name = BREED_CN.get(name, name)
        print(f"  {cn_name:<12}  {pc['precision']*100:>6.2f}%  {pc['recall']*100:>6.2f}%  "
              f"{pc['f1']:>6.4f}  {pc['accuracy']*100:>6.2f}%  {pc['total']:>5}")

    # 推理延迟
    lat = results["latency"]
    print(f"\n⏱️  推理延迟（{results['dataset_size']} 张图片）")
    print(f"  {'─' * 40}")
    print(f"  平均延迟:      {lat['mean_ms']:.2f} ms / 张")
    print(f"  中位延迟:      {lat['median_ms']:.2f} ms / 张")
    print(f"  P95 延迟:      {lat['p95_ms']:.2f} ms / 张")
    print(f"  吞吐量:        {lat['fps']:.1f} FPS")

    if lat['p95_ms'] < 10:
        print(f"\n  ✅ P95 延迟 < 10ms，满足毫秒级实时推理要求！")
    elif lat['p95_ms'] < 50:
        print(f"\n  ⚠️  P95 延迟 < 50ms，在可接受范围内。")
    else:
        print(f"\n  ❌ P95 延迟 > 50ms，建议进一步优化（FP16/TensorRT 等）。")

    # 混淆矩阵
    print(f"\n🔍 混淆矩阵（行 = 真实标签，列 = 预测标签）")
    print(f"  {'─' * 60}")
    cm = build_confusion_matrix(
        results["_predictions"], results["_labels"], results["num_classes"]
    )
    # 表头
    short_names = [BREED_CN.get(n, n)[:3] for n in class_names]
    header = "           " + "".join(f"{n:>7}" for n in short_names)
    print(f"  真实↓预测→{header}")
    for i, name in enumerate(class_names):
        cn_name = BREED_CN.get(name, name)
        row = "".join(f"{cm[i][j]:>7}" for j in range(len(class_names)))
        print(f"  {cn_name:<10} {row}")

    print(f"\n{'=' * 70}")


def save_results(results: dict, output_path: str):
    """将评估结果保存为 JSON 文件（不包含 _predictions/_labels 原始数据）。"""
    # 去除原始数据字段
    clean_results = {k: v for k, v in results.items()
                     if not k.startswith("_")}
    clean_results["confusion_matrix"] = build_confusion_matrix(
        results["_predictions"], results["_labels"], results["num_classes"]
    ).tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {output_path}")


# ============================================================
# 单张图片推理
# ============================================================

@torch.no_grad()
def predict_single(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    use_half: bool,
    class_names: list[str],
    input_size: int = 224,
):
    """对单张图片进行推理并输出结果。"""
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    if use_half:
        input_tensor = input_tensor.half()

    model = model.to(device)
    if use_half:
        model = model.half()

    t0 = time.perf_counter()
    outputs = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())

    print(f"\n{'=' * 50}")
    print(f"图片: {image_path}")
    print(f"推理耗时: {elapsed_ms:.2f} ms")
    print(f"\n预测结果（Top-5）:")
    top5 = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5):
        name = class_names[idx]
        cn_name = BREED_CN.get(name, name)
        marker = " ← 最佳预测" if rank == 0 else ""
        print(f"  {rank+1}. {cn_name} ({name}): {probs[idx]*100:.2f}%{marker}")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="评估 ResNet-50 猫品种识别模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 在独立测试集上评估
  python evaluate_cnn.py --model best_model.pth --test-list outputs/.../test_image_paths.txt

  # 单张图片推理
  python evaluate_cnn.py --model best_model.pth --image cat.jpg

  # 评估并保存 JSON 结果
  python evaluate_cnn.py --model best_model.pth --output results.json
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="模型检查点路径 (.pth)")
    parser.add_argument("--test-list", type=str, default=None,
                        help="测试集图片路径清单（train_cnn.py 自动生成）")
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片路径（单图推理模式）")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果 JSON 输出路径")
    parser.add_argument("--batch", type=int, default=32,
                        help="评估批次大小（默认 32）")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="类别数（默认 5）")
    parser.add_argument("--no-half", action="store_true",
                        help="禁用 FP16 推理")
    args = parser.parse_args()

    # ---- 单张图片推理模式 ----
    if args.image:
        device, use_half = get_device(use_half=not args.no_half)
        model, input_size, class_names = load_model(args.model, num_classes=args.num_classes)
        predict_single(model, args.image, device, use_half, class_names, input_size)
        return

    # ---- 测试集评估模式 ----
    device, use_half = get_device(use_half=not args.no_half)
    model, input_size, class_names = load_model(args.model, num_classes=args.num_classes)

    # 构建测试集
    if args.test_list:
        test_dataset = TestImageDataset(args.test_list, input_size=input_size)
    else:
        logger.error(
            "请指定测试集路径清单。方式：\n"
            "  1. 训练时使用 --test-ratio 0.1，自动生成 test_image_paths.txt\n"
            "  2. 手动指定: --test-list <path>\n"
            "  3. 单张推理: --image <path>"
        )
        sys.exit(1)

    # 类别名优先使用 checkpoint 中的，否则从测试集清单推断
    if not class_names:
        first_sample = test_dataset.samples[0]
        class_names_set = sorted(set(s[2] for s in test_dataset.samples if s[2]))
        class_names = class_names_set if class_names_set else [f"class_{i}" for i in range(args.num_classes)]

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 评估
    results = evaluate(model, test_loader, device, use_half, class_names)

    # 输出
    print_results(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
