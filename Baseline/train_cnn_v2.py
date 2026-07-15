"""
猫品种识别模型 — ConvNeXt / EfficientNetV2 迁移学习训练脚本 (V2)
================================================================

相比 train_cnn.py (ResNet-50)，本脚本针对 RTX 4070 (12GB) 做了以下升级：

1. **现代骨干网络** — 默认使用 ConvNeXt-Tiny，可选 ConvNeXt-Small / EfficientNetV2-S
   - ConvNeXt: 融合 CNN 与 ViT 设计理念，细粒度分类 SOTA
   - EfficientNetV2: 训练更快、精度更高，参数效率极佳

2. **更大输入分辨率** — 384×384（ConvNeXt 原生分辨率），提取更精细特征

3. **更强的正则化** — MixUp + CutMix + Stochastic Depth，在小数据集上抑制过拟合

4. **Cosine Warmup 学习率调度** — 前 5 个 epoch 线性预热，避免初期不稳定

5. **优化 RTX 4070 配置** — batch_size=64, input_size=384, 更深层解冻

用法：
    python train_cnn_v2.py                                    # 默认 ConvNeXt-Tiny
    python train_cnn_v2.py --backbone convnext_small          # ConvNeXt-Small
    python train_cnn_v2.py --backbone efficientnet_v2_s       # EfficientNetV2-S
    python train_cnn_v2.py --backbone resnet50                # ResNet-50（向后兼容）
    python train_cnn_v2.py --data ./my_images --epochs 100    # 自定义参数
    python train_cnn_v2.py --resume checkpoints/xxx.pth       # 从检查点恢复
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_v2.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# MixUp & CutMix 数据增强
# ============================================================

def rand_bbox(size, lam):
    """生成 CutMix 的随机矩形区域。"""
    W, H = size[-2], size[-1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_cutmix_collate(batch, alpha: float = 0.8, cutmix_prob: float = 0.5):
    """
    MixUp / CutMix 混合 collate 函数。

    对每个 batch 以一定概率应用 MixUp 或 CutMix：
    - MixUp: 两张图像按 λ 线性混合，标签也按 λ 混合
    - CutMix: 将一张图像的矩形区域粘贴到另一张上

    Args:
        batch: DataLoader 返回的原始 batch
        alpha: Beta 分布参数（越大混合越强）
        cutmix_prob: 使用 CutMix 而非 MixUp 的概率

    Returns:
        (mixed_images, labels_a, labels_b, lam)
    """
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    if np.random.random() > 0.5:
        # 不应用混合，保持原样
        return images, labels, labels, torch.tensor(1.0)

    lam = np.random.beta(alpha, alpha)

    # 随机决定用 CutMix 还是 MixUp
    if np.random.random() < cutmix_prob:
        # CutMix
        idx = torch.randperm(images.size(0))
        x1, y1, x2, y2 = rand_bbox(images.shape, lam)
        images[:, :, y1:y2, x1:x2] = images[idx][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        # MixUp
        idx = torch.randperm(images.size(0))
        images = lam * images + (1.0 - lam) * images[idx]
        # 注意: labels_a 是原始标签, labels_b 是混合标签
        labels_a = labels
        labels_b = labels[idx]

    return images, labels, labels[idx], torch.tensor(lam)


# ============================================================
# GPU 检测与配置
# ============================================================

def detect_gpu() -> dict:
    """检测 GPU 设备并返回配置信息。"""
    gpu_info: dict = {
        "device": torch.device("cpu"),
        "name": "CPU",
        "vram_gb": 0,
        "use_amp": False,
        "cuda_available": False,
    }

    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA 不可用，将使用 CPU 训练（速度会很慢）")
        return gpu_info

    gpu_info["cuda_available"] = True
    device_name = torch.cuda.get_device_name(0)
    gpu_info["name"] = device_name

    try:
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_info["vram_gb"] = round(vram_bytes / (1024 ** 3), 1)
    except Exception:
        gpu_info["vram_gb"] = 4.0

    gpu_info["device"] = torch.device("cuda:0")
    gpu_info["use_amp"] = True

    # cuDNN 自动调优
    torch.backends.cudnn.benchmark = True
    # Ada Lovelace (RTX 4070) 额外优化
    if gpu_info["vram_gb"] >= 10:
        torch.set_float32_matmul_precision("high")

    logger.info("=" * 60)
    logger.info("GPU 检测报告")
    logger.info("=" * 60)
    logger.info(f"  设备名称:     {device_name}")
    logger.info(f"  显存大小:     {gpu_info['vram_gb']} GB")
    logger.info(f"  CUDA 版本:    {torch.version.cuda}")
    logger.info(f"  PyTorch 版本: {torch.__version__}")
    logger.info(f"  混合精度训练: {'✅ 启用' if gpu_info['use_amp'] else '❌ 不支持'}")
    logger.info("=" * 60)

    return gpu_info


# ============================================================
# 数据集与数据增强
# ============================================================

# 类别列表 — 注意：ImageFolder 按目录名字母排序，最终顺序由 class_info 决定
# 实际顺序（字母序）: other(如果有), pallas, persian, ragdoll, singapura, sphynx
# 如需引入"非猫"类别，创建 other/ 目录并放入随机非猫图片即可自动识别
CAT_BREEDS_SORTED = ["pallas", "persian", "ragdoll", "singapura", "sphynx"]

BREED_CN = {
    "pallas": "兔狲",
    "persian": "波斯猫",
    "ragdoll": "布偶猫",
    "singapura": "新加坡猫",
    "sphynx": "斯芬克斯猫",
    "other": "非猫/其他",
}


def get_transforms(input_size: int = 384) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    构建训练集和验证集的图像预处理变换。

    训练集使用增强的数据增强策略：
    - RandAugment 风格的自动增强
    - RandomErasing 随机遮挡
    - 更强的颜色抖动和几何变换

    验证集仅使用中心裁剪 + 归一化。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 训练集增强 — 针对细粒度分类的强化策略
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(
            brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3),
        ),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transforms, val_transforms


def prepare_datasets(
    data_root: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    seed: int = 42,
    input_size: int = 384,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, Optional[datasets.ImageFolder], dict]:
    """准备训练集、验证集和可选的独立测试集。"""
    from torch.utils.data import Subset, random_split

    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据目录不存在: {data_path}\n"
            f"请先运行以下脚本搜集图片：\n"
            f"  猫品种图片: image_collector/collect_cat_images.py\n"
            f"  随机非猫图片: image_collector/collect_random_images.py"
        )

    train_transforms, val_transforms = get_transforms(input_size)
    full_dataset = datasets.ImageFolder(str(data_path), transform=None)
    total_size = len(full_dataset)

    logger.info("\n数据集目录结构:")
    # 使用 ImageFolder 实际发现的类别（字母序），而非硬编码列表
    actual_classes = full_dataset.classes
    unknown_dirs = [c for c in actual_classes
                    if c not in CAT_BREEDS_SORTED and c != "other"]
    if unknown_dirs:
        logger.warning(
            f"  ⚠️  发现未知目录将被作为独立类别训练: {unknown_dirs}\n"
            f"     如果这是意外，请删除或合并这些目录后重新训练！"
        )
    for breed in actual_classes:
        breed_dir = data_path / breed
        count = len(list(breed_dir.glob("*.[jJ][pP][gG]")))
        count += len(list(breed_dir.glob("*.[pP][nN][gG]")))
        cn_name = BREED_CN.get(breed, breed)
        tag = " 🐱" if breed in CAT_BREEDS_SORTED else (" 🚫 非猫拒识类" if breed == "other" else " ❓ 未知类")
        logger.info(f"  {breed}/ ({cn_name}): {count} 张{tag}")

    generator = torch.Generator().manual_seed(seed)
    test_dataset = None
    test_indices: list[int] = []

    if test_ratio > 0:
        test_size = int(total_size * test_ratio)
        remain_size = total_size - test_size
        remain_subset, test_subset = random_split(
            full_dataset, [remain_size, test_size], generator=generator,
        )
        remain_indices = remain_subset.indices
        test_indices = list(test_subset.indices)
        logger.info(f"\n  先划分测试集: {test_size} 张 ({test_ratio*100:.0f}%)")
    else:
        remain_indices = list(range(total_size))

    remain_size = len(remain_indices)
    val_size = int(remain_size * val_ratio)
    train_size = remain_size - val_size

    perm = torch.randperm(remain_size, generator=generator).tolist()
    train_local = perm[:train_size]
    val_local = perm[train_size:]
    train_indices = [remain_indices[i] for i in train_local]
    val_indices = [remain_indices[i] for i in val_local]

    train_full = datasets.ImageFolder(str(data_path), transform=train_transforms)
    val_full = datasets.ImageFolder(str(data_path), transform=val_transforms)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    if test_ratio > 0:
        test_full = datasets.ImageFolder(str(data_path), transform=val_transforms)
        test_dataset = Subset(test_full, test_indices)

    class_info = {
        "classes": full_dataset.classes,
        "num_classes": len(full_dataset.classes),
        "class_to_idx": full_dataset.class_to_idx,
    }

    logger.info(f"\n数据集划分:")
    logger.info(f"  总图片数: {total_size}")
    if test_ratio > 0:
        logger.info(f"  测试集:   {len(test_indices)} ({len(test_indices)/total_size*100:.1f}%)")
    logger.info(f"  训练集:   {train_size} ({train_size/total_size*100:.1f}%)")
    logger.info(f"  验证集:   {val_size} ({val_size/total_size*100:.1f}%)")
    logger.info(f"  类别数:   {class_info['num_classes']}")

    return train_dataset, val_dataset, test_dataset, class_info


# ============================================================
# 模型构建 — 支持多种骨干网络
# ============================================================

# 骨干网络配置表
BACKBONE_CONFIGS = {
    "convnext_tiny": {
        "name": "ConvNeXt-Tiny",
        "input_size": 384,
        "weight_name": "IMAGENET1K_V1",
        "num_features": 768,
        "dropout_p": 0.5,
        "description": "现代CNN架构（2022），融合ViT设计理念。轻量但精度高，推荐首选。",
    },
    "convnext_small": {
        "name": "ConvNeXt-Small",
        "input_size": 384,
        "weight_name": "IMAGENET1K_V1",
        "num_features": 768,
        "dropout_p": 0.6,
        "description": "ConvNeXt中等规模，~50M参数，RTX 4070 12GB 可轻松驾驭。细粒度分类最强。",
    },
    "convnext_base": {
        "name": "ConvNeXt-Base",
        "input_size": 384,
        "weight_name": "IMAGENET1K_V1",
        "num_features": 1024,
        "dropout_p": 0.6,
        "description": "ConvNeXt大规模，~89M参数。需要12GB+显存，精度天花板。",
    },
    "efficientnet_v2_s": {
        "name": "EfficientNetV2-S",
        "input_size": 384,
        "weight_name": "IMAGENET1K_V1",
        "num_features": 1280,
        "dropout_p": 0.5,
        "description": "Google 2021年出品，训练速度极快，参数效率最高。",
    },
    "efficientnet_v2_m": {
        "name": "EfficientNetV2-M",
        "input_size": 384,
        "weight_name": "IMAGENET1K_V1",
        "num_features": 1280,
        "dropout_p": 0.6,
        "description": "EfficientNetV2中等规模，~54M参数。精度与速度的平衡点。",
    },
    "resnet50": {
        "name": "ResNet-50",
        "input_size": 224,
        "weight_name": "IMAGENET1K_V2",
        "num_features": 2048,
        "dropout_p": 0.5,
        "description": "经典骨干网络，向后兼容。",
    },
}

# ConvNeXt 各阶段对应的 layer 名称（用于差异化解冻）
CONVNEXT_STAGES = ["features.4", "features.5", "features.6", "features.7"]
EFFICIENTNET_STAGES = ["features.5", "features.6", "features.7"]


def build_model(
    backbone: str = "convnext_tiny",
    num_classes: int = 5,
    freeze_backbone: bool = True,
    dropout_p: Optional[float] = None,
) -> Tuple[nn.Module, int]:
    """
    构建基于指定骨干网络的迁移学习模型。

    支持的骨干网络：
    - convnext_tiny / convnext_small / convnext_base
    - efficientnet_v2_s / efficientnet_v2_m
    - resnet50

    Args:
        backbone: 骨干网络名称
        num_classes: 类别数
        freeze_backbone: 是否冻结骨干网络
        dropout_p: 分类头 Dropout 比例（None 使用默认值）

    Returns:
        (model, input_size)
    """
    from torchvision import models

    config = BACKBONE_CONFIGS[backbone]
    model_name = config["name"]
    input_size = config["input_size"]
    num_features = config["num_features"]
    dp = dropout_p if dropout_p is not None else config["dropout_p"]

    logger.info(f"\n构建模型: {model_name} (预训练于 ImageNet-1K)")
    logger.info(f"  骨干网络:    {backbone}")
    logger.info(f"  输入分辨率:  {input_size}×{input_size}")
    logger.info(f"  特征维度:    {num_features}")
    logger.info(f"  输出类别数:  {num_classes}")

    # ---- 加载预训练模型 ----
    if backbone.startswith("convnext"):
        pretrained_weights = getattr(models, f"ConvNeXt_{backbone.split('_')[1].capitalize()}_Weights").IMAGENET1K_V1
        model = getattr(models, f"convnext_{backbone.split('_')[1]}")(weights=pretrained_weights)
        # ConvNeXt 的分类头在 classifier 中
        classifier_in = model.classifier[2].in_features
        head_attr = "classifier"
        stages = CONVNEXT_STAGES

    elif backbone.startswith("efficientnet_v2"):
        pretrained_weights = getattr(
            models, f"EfficientNet_V2_{backbone.split('_')[2].upper()}_Weights"
        ).IMAGENET1K_V1
        model = getattr(models, f"efficientnet_v2_{backbone.split('_')[2]}")(weights=pretrained_weights)
        classifier_in = model.classifier[1].in_features
        head_attr = "classifier"
        stages = EFFICIENTNET_STAGES

    elif backbone == "resnet50":
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        classifier_in = model.fc.in_features
        head_attr = "fc"
        stages = ["layer2", "layer3", "layer4"]

    else:
        raise ValueError(f"不支持的骨干网络: {backbone}")

    # ---- 替换分类头 ----
    # 为细粒度分类设计多层分类头：更强的非线性 + Dropout 防过拟合
    # 注意：ConvNeXt 的 classifier 内部包含 Flatten，替换后需补上；
    #       对 ResNet/EfficientNet（已在 forward 中 flatten），Flatten 是无操作的。
    new_head = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(classifier_in, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dp),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dp * 0.8),
        nn.Linear(512, num_classes),
    )

    # 对 Linear 层做 Kaiming 初始化
    for m in new_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    # 替换分类头
    if head_attr == "classifier":
        model.classifier = new_head
    elif head_attr == "fc":
        model.fc = new_head

    # ---- 冻结 / 解冻 ----
    if freeze_backbone:
        logger.info(f"  骨干网络: 冻结 ❄️（仅训练分类头）")
        head_params = set()
        head_module = getattr(model, head_attr)
        for p in head_module.parameters():
            head_params.add(id(p))

        for param in model.parameters():
            if id(param) not in head_params:
                param.requires_grad = False
    else:
        logger.info(f"  骨干网络: 可训练 🔥")

    # 统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    return model, input_size


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMAModel:
    """指数移动平均模型参数平滑器。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ============================================================
# 训练与验证函数
# ============================================================

def mixup_loss(criterion, outputs, labels_a, labels_b, lam):
    """MixUp/CutMix 混合损失。"""
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    total_epochs: int,
    max_grad_norm: float = 1.0,
    use_mixup: bool = True,
    mixup_alpha: float = 0.8,
) -> Tuple[float, float]:
    """训练一个 epoch。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp / CutMix
        if use_mixup and np.random.random() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(images.size(0))
            if np.random.random() < 0.5:
                # CutMix
                x1, y1, x2, y2 = rand_bbox(images.shape, lam)
                images[:, :, y1:y2, x1:x2] = images[idx][:, :, y1:y2, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
            else:
                # MixUp
                images = lam * images + (1.0 - lam) * images[idx]

            labels_a, labels_b = labels, labels[idx]
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                outputs = model(images)
                if mixed:
                    loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if mixed:
                loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item()
        if not mixed:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) * dataloader.batch_size / elapsed
            logger.info(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {running_loss/(batch_idx+1):.4f} "
                f"Acc: {correct/max(total,1)*100:.2f}% "
                f"Speed: {speed:.0f} img/s"
            )

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """在验证集上评估模型。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), correct / total


# ============================================================
# Cosine Warmup 学习率调度器
# ============================================================

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine 退火 + 线性预热学习率调度器。"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 1e-3,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            scale = (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine 退火
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================
# 主训练流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ConvNeXt / EfficientNetV2 猫品种识别训练脚本 (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模型参数
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="骨干网络（默认 convnext_tiny）。\n"
             "  convnext_tiny      - 现代CNN，轻量高效（推荐）\n"
             "  convnext_small     - 更大模型，精度更高（12GB推荐）\n"
             "  convnext_base      - 最大模型，需要12GB+\n"
             "  efficientnet_v2_s  - 训练极快，参数效率高\n"
             "  efficientnet_v2_m  - 均衡精度与速度\n"
             "  resnet50           - 经典ResNet（向后兼容）",
    )

    # 数据参数
    parser.add_argument("--data", type=str, default="image_collector/collected_images",
                        help="数据集根目录路径")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="独立测试集比例")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=80,
                        help="阶段二微调总轮数（默认 80）")
    parser.add_argument("--phase1-epochs", type=int, default=20,
                        help="阶段一特征提取的轮数（默认 20）")
    parser.add_argument("--batch", type=int, default=None,
                        help="批次大小（默认根据GPU自动选择）")
    parser.add_argument("--input-size", type=int, default=None,
                        help="输入分辨率（默认使用骨干网络原生分辨率）")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="阶段一初始学习率")
    parser.add_argument("--lr-finetune", type=float, default=0.0001,
                        help="阶段二微调学习率")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="学习率预热轮数（默认 5）")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label Smoothing 系数")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="权重衰减系数")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--mixup-alpha", type=float, default=0.8,
                        help="MixUp/CutMix 的 Beta 分布参数")
    parser.add_argument("--dropout", type=float, default=None,
                        help="分类头 Dropout 比例（None 使用默认值）")

    # 其他
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--output", type=str, default="outputs",
                        help="输出目录")
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--no-mixup", action="store_true",
                        help="禁用 MixUp/CutMix 数据增强")

    args = parser.parse_args()

    # ============================================================
    # 0. 初始化
    # ============================================================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    config = BACKBONE_CONFIGS[args.backbone]

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.backbone.replace("_", "")
    output_dir = Path(args.output) / f"{model_tag}_cat_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # ============================================================
    # 1. GPU 检测 & 自动调优
    # ============================================================
    gpu_info = detect_gpu()
    device = gpu_info["device"]
    use_amp = gpu_info["use_amp"] and not args.no_amp

    # 输入分辨率
    if args.input_size is None:
        args.input_size = config["input_size"]
        logger.info(f"使用骨干网络原生分辨率: {args.input_size}")

    # 批次大小自动调优
    if args.batch is None:
        vram_gb = gpu_info.get("vram_gb", 0)
        if vram_gb >= 16:
            args.batch = 96
        elif vram_gb >= 12:
            args.batch = 64
        elif vram_gb >= 8:
            args.batch = 32
        elif vram_gb >= 4:
            args.batch = 16
        else:
            args.batch = 8
        logger.info(f"自动选择 batch_size={args.batch}（显存 {vram_gb}GB）")

    # workers 自动调优
    if args.workers == 4:
        suggested = min(os.cpu_count() or 4, 12)
        if suggested > args.workers:
            args.workers = suggested

    # 大显存进一步优化
    vram_gb = gpu_info.get("vram_gb", 0)
    if vram_gb >= 10:
        logger.info(f"\n🔧 检测到大显存 GPU ({vram_gb} GB)，启用优化配置")
        logger.info(f"   骨干网络:     {config['name']}")
        logger.info(f"   输入分辨率:   {args.input_size}×{args.input_size}")
        logger.info(f"   batch_size:   {args.batch}")
        logger.info(f"   workers:      {args.workers}")
        logger.info(f"   MixUp/CutMix: {'✅' if not args.no_mixup else '❌ 已禁用'}")
        logger.info(f"   FP16:         {'✅' if use_amp else '❌'}")

    # ============================================================
    # 2. 准备数据集
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("准备数据集")
    logger.info("=" * 60)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        script_dir = Path(__file__).resolve().parent
        data_root = script_dir / args.data
    logger.info(f"数据路径: {data_root}")

    train_dataset, val_dataset, test_dataset, class_info = prepare_datasets(
        str(data_root),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        input_size=args.input_size,
    )

    # 保存测试集路径
    if test_dataset is not None:
        test_paths_file = output_dir / "test_image_paths.txt"
        full_dataset = datasets.ImageFolder(str(data_root))
        with open(test_paths_file, "w", encoding="utf-8") as f:
            for idx in sorted(test_dataset.indices):
                img_path, label = full_dataset.imgs[idx]
                f.write(f"{img_path}\t{label}\t{full_dataset.classes[label]}\n")
        logger.info(f"测试集路径清单已保存: {test_paths_file}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if gpu_info["cuda_available"] else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if gpu_info["cuda_available"] else False,
    )

    num_classes = class_info["num_classes"]
    use_mixup = not args.no_mixup

    # ============================================================
    # 3. 构建模型
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("构建模型")
    logger.info("=" * 60)

    model, _ = build_model(
        backbone=args.backbone,
        num_classes=num_classes,
        freeze_backbone=True,
        dropout_p=args.dropout,
    )
    model = model.to(device)

    # ============================================================
    # 4. 训练
    # ============================================================
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler() if use_amp else None

    logger.info(f"\n正则化配置:")
    logger.info(f"  Label Smoothing:     {args.label_smoothing}")
    logger.info(f"  Weight Decay:        {args.weight_decay}")
    logger.info(f"  Gradient Clipping:   {args.grad_clip}")
    logger.info(f"  MixUp/CutMix α:      {args.mixup_alpha}" if use_mixup else "  MixUp/CutMix:        ❌ 已禁用")
    logger.info(f"  Dropout:             {args.dropout or config['dropout_p']}")
    logger.info(f"  Warmup Epochs:       {args.warmup_epochs}")

    best_val_acc = 0.0
    total_epoch = 0
    ema: Optional[EMAModel] = None

    # ----------------------------------------------------------
    # 阶段一：特征提取
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("阶段一：特征提取（Feature Extraction）")
    logger.info(f"  - 骨干网络: 冻结 ❄️")
    logger.info(f"  - 分类头:   训练 🔥")
    logger.info(f"  - 学习率:   {args.lr}（+ {args.warmup_epochs} epoch 预热）")
    logger.info(f"  - Epochs:   1-{args.phase1_epochs}")
    logger.info("=" * 60)

    head_attr = "classifier" if args.backbone.startswith(("convnext", "efficientnet")) else "fc"
    head_params = getattr(model, head_attr).parameters()
    optimizer = optim.AdamW(
        head_params, lr=args.lr, weight_decay=args.weight_decay * 0.2,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.phase1_epochs,
    )

    for epoch in range(1, args.phase1_epochs + 1):
        total_epoch += 1
        logger.info(f"\n--- Phase 1 Epoch {epoch}/{args.phase1_epochs} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, args.phase1_epochs,
            max_grad_norm=args.grad_clip,
            use_mixup=use_mixup,
            mixup_alpha=args.mixup_alpha,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
                     f" | LR: {current_lr:.6f}")

        writer.add_scalar("Phase1/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase1/Train_Acc", train_acc, epoch)
        writer.add_scalar("Phase1/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase1/Val_Acc", val_acc, epoch)
        writer.add_scalar("Phase1/LR", current_lr, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_phase1.pth"
            torch.save({
                "epoch": epoch,
                "backbone": args.backbone,
                "input_size": args.input_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
            }, checkpoint_path)
            logger.info(f"  ✅ 最佳模型已保存: {checkpoint_path}")

    logger.info(f"\n阶段一完成！最佳验证准确率: {best_val_acc*100:.2f}%")

    # ----------------------------------------------------------
    # 阶段二：微调
    # ----------------------------------------------------------
    unfreeze_stages = 2 if vram_gb >= 10 else 1  # 大显存解冻更多层

    logger.info("\n" + "=" * 60)
    logger.info("阶段二：微调（Fine-tuning）")
    logger.info(f"  - 解冻层数:          {unfreeze_stages}（大显存模式）" if vram_gb >= 10 else "  - 解冻层数:          1")
    logger.info("  - 分类头:           训练 🔥")
    logger.info(f"  - 学习率:           差异化 LR")
    logger.info(f"  - EMA:               decay=0.999 ✅")
    logger.info("=" * 60)

    # 加载阶段一最佳权重
    best_phase1_path = checkpoint_dir / "best_phase1.pth"
    if best_phase1_path.exists():
        checkpoint = torch.load(best_phase1_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载阶段一最佳模型: val_acc={checkpoint['val_acc']*100:.2f}%")

    # 解冻策略：对于 ConvNeXt，stages 末尾几层
    # features.4 → stem + 下采样、features.5-7 → 中高层特征
    if args.backbone.startswith("convnext"):
        # 解冻最后 unfreeze_stages 个 stage
        stages_to_unfreeze = CONVNEXT_STAGES[-unfreeze_stages - 1:]
        for name, param in model.named_parameters():
            if any(stage in name for stage in stages_to_unfreeze):
                param.requires_grad = True
    elif args.backbone.startswith("efficientnet"):
        stages_to_unfreeze = EFFICIENTNET_STAGES[-unfreeze_stages:]
        for name, param in model.named_parameters():
            if any(stage in name for stage in stages_to_unfreeze):
                param.requires_grad = True
    else:
        # ResNet
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            elif "layer2" in name and vram_gb >= 10:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # 差异化学习率
    optimizer = optim.AdamW([
        {
            "params": getattr(model, head_attr).parameters(),
            "lr": args.lr_finetune * 3,
            "weight_decay": args.weight_decay * 0.5,
        },
        {
            "params": (p for n, p in model.named_parameters()
                        if "features.7" in n or "layer4" in n),
            "lr": args.lr_finetune * 0.5,
            "weight_decay": args.weight_decay,
        },
        {
            "params": (p for n, p in model.named_parameters()
                        if ("features.6" in n or "features.5" in n or
                            "features.4" in n or "layer3" in n or "layer2" in n)
                        and p.requires_grad),
            "lr": args.lr_finetune * 0.2,
            "weight_decay": args.weight_decay,
        },
    ])

    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=min(args.warmup_epochs, 3),
        total_epochs=args.epochs - args.phase1_epochs,
    )

    ema = EMAModel(model, decay=0.999)
    logger.info("EMA 已初始化 (decay=0.999)")

    patience = 10 if vram_gb >= 10 else 7
    no_improve = 0
    best_val_acc_phase2 = best_val_acc
    best_phase2_val = 0.0

    phase2_epochs = args.epochs - args.phase1_epochs

    for epoch in range(1, phase2_epochs + 1):
        total_epoch += 1
        logger.info(f"\n--- Phase 2 Epoch {epoch}/{phase2_epochs} "
                     f"(Total: {total_epoch}/{args.epochs}) ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, phase2_epochs,
            max_grad_norm=args.grad_clip,
            use_mixup=use_mixup,
            mixup_alpha=args.mixup_alpha,
        )

        ema.update(model)

        ema.apply_shadow(model)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        ema.restore(model)

        current_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()

        acc_gap = train_acc - val_acc
        gap_warning = ""
        if acc_gap > 0.10:
            gap_warning = f" ⚠️ 过拟合严重！(gap={acc_gap*100:.1f}%)"
        elif acc_gap > 0.05:
            gap_warning = f" ⚡ 轻微过拟合 (gap={acc_gap*100:.1f}%)"

        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
                     f" | LRs: {[f'{lr:.6f}' for lr in current_lrs]}"
                     f" | Gap: {acc_gap*100:.1f}%{gap_warning}")

        writer.add_scalar("Phase2/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase2/Train_Acc", train_acc, epoch)
        writer.add_scalar("Phase2/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase2/Val_Acc", val_acc, epoch)
        writer.add_scalar("Phase2/TrainVal_Gap", acc_gap, epoch)

        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            no_improve = 0
            checkpoint_path = checkpoint_dir / "best_model.pth"
            ema.apply_shadow(model)
            torch.save({
                "epoch": total_epoch,
                "backbone": args.backbone,
                "input_size": args.input_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
                "phase": "phase2",
            }, checkpoint_path)
            ema.restore(model)
            logger.info(f"  ✅ 最佳模型已保存（EMA 权重）: {checkpoint_path}")
        else:
            no_improve += 1
            logger.info(f"  未提升 ({no_improve}/{patience})")

        if val_acc > best_phase2_val:
            best_phase2_val = val_acc
            ema.apply_shadow(model)
            torch.save({
                "epoch": total_epoch,
                "backbone": args.backbone,
                "input_size": args.input_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
                "phase": "phase2",
            }, checkpoint_dir / "best_phase2.pth")
            ema.restore(model)

        if no_improve >= patience:
            logger.info(f"\n早停触发！验证准确率连续 {patience} 轮未提升")
            break

    # ============================================================
    # 4.5 保证 best_model.pth 存在
    # ============================================================
    if not (checkpoint_dir / "best_model.pth").exists():
        logger.warning("⚠️  Phase2 未超越 Phase1，从可用 checkpoint 中选择最佳模型")
        candidates = []
        for ckpt_name in ["best_phase2.pth", "best_phase1.pth"]:
            ckpt_path = checkpoint_dir / ckpt_name
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                candidates.append((ckpt_name, ckpt.get("val_acc", 0.0)))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            src_name, src_val = candidates[0]
            src = checkpoint_dir / src_name
            ckpt = torch.load(src, map_location="cpu")
            dst = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": ckpt.get("epoch", total_epoch),
                "backbone": args.backbone,
                "input_size": args.input_size,
                "model_state_dict": ckpt.get("model_state_dict", ckpt),
                "optimizer_state_dict": ckpt.get("optimizer_state_dict", {}),
                "val_acc": ckpt.get("val_acc", 0.0),
                "class_info": class_info,
                "phase": f"{src_name}_fallback",
            }, dst)
            logger.info(f"  📦 将 {src_name} (val_acc={src_val*100:.2f}%) 复制为 best_model.pth")

    # ============================================================
    # 5. 最终输出
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"骨干网络:       {config['name']}")
    logger.info(f"最佳验证准确率: {best_val_acc_phase2*100:.2f}%")
    logger.info(f"总训练轮数:     {total_epoch}")
    logger.info(f"模型保存路径:   {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"TensorBoard:    tensorboard --logdir {log_dir}")

    # 导出 TorchScript
    try:
        best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()
        example_input = torch.randn(1, 3, args.input_size, args.input_size)
        traced_model = torch.jit.trace(model.cpu(), example_input)
        script_path = output_dir / f"{model_tag}_cat_scripted.pt"
        traced_model.save(str(script_path))
        logger.info(f"TorchScript 模型: {script_path}")
    except Exception as e:
        logger.warning(f"TorchScript 导出失败: {e}")

    # 导出 ONNX
    try:
        model = model.to("cpu")
        model.eval()
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
        onnx_path = output_dir / f"{model_tag}_cat.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"ONNX 模型:      {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX 导出失败: {e}")

    writer.close()

    logger.info(f"\n{'=' * 60}")
    logger.info("输出文件汇总")
    logger.info(f"{'=' * 60}")
    logger.info(f"  最佳模型:     {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"  训练日志:     training_v2.log")
    logger.info(f"  TensorBoard:  {log_dir}")
    logger.info(f"")
    logger.info(f"下一步:")
    logger.info(f"  1. 查看训练曲线: tensorboard --logdir {log_dir}")
    logger.info(f"  2. 评估模型:     python evaluate_cnn.py --model {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
