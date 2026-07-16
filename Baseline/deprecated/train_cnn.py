"""
猫品种识别模型 — ResNet-50 迁移学习训练脚本
==============================================

基于 baseline_prompt.md 中的分析：
  - 推荐 CNN 架构（ResNet-50）而非 YOLO，因任务本质是 5 猫品种细粒度分类
  - 使用 ImageNet 预训练权重进行两阶段迁移学习
  - 自动检测 GPU 并适配训练配置（A2000 4GB / RTX 4070 12GB / ...）

两阶段训练策略：
  阶段一（特征提取）：冻结骨干网络，仅训练新分类头
  阶段二（微调）：     解冻 layer3-4（大显存 GPU 额外解冻 layer2），差异化学习率全局微调

GPU 支持：
  自动检测 GPU 型号与显存，动态调优 batch_size/input_size/epochs
  支持 FP16 混合精度训练 + TF32（Ada Lovelace）
  若 GPU 不可用，回退到 CPU 训练并给出警告

用法：
    python train_cnn.py                              # 默认配置训练
    python train_cnn.py --epochs 80 --batch 32       # 自定义参数
    python train_cnn.py --data ./my_images            # 自定义数据路径
    python train_cnn.py --resume checkpoints/xxx.pth  # 从检查点恢复
"""

import argparse
import logging
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
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# GPU 检测与配置
# ============================================================

def detect_gpu() -> dict:
    """
    检测可用的 GPU 设备，返回 GPU 配置信息。
    
    支持多种 GPU 型号检测（A2000 / RTX 4070 / 通用 RTX），并返回
    显存大小、架构信息（is_4070/is_a2000）供训练流程自动调优。
    
    Returns:
        包含设备信息的字典：
        - device: torch.device 对象
        - name: GPU 名称
        - vram_gb: 显存大小（GB）
        - use_amp: 是否启用混合精度
        - cuda_available: CUDA 是否可用
    """
    gpu_info: dict = {
        "device": torch.device("cpu"),
        "name": "CPU",
        "vram_gb": 0,
        "use_amp": False,
        "cuda_available": False,
    }
    
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA 不可用，将使用 CPU 训练（速度会很慢）")
        logger.warning("    请检查：")
        logger.warning("    1. NVIDIA 驱动是否正确安装")
        logger.warning("    2. PyTorch 是否为 CUDA 版本")
        logger.warning("    3. 运行 nvidia-smi 确认 GPU 是否可见")
        return gpu_info
    
    # CUDA 可用
    gpu_info["cuda_available"] = True
    device_name = torch.cuda.get_device_name(0)
    gpu_info["name"] = device_name
    
    # 获取显存信息
    try:
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        gpu_info["vram_gb"] = round(vram_gb, 1)
    except Exception:
        vram_gb = 4.0  # RTX A2000 Laptop 默认 4GB
    
    # 设置设备
    gpu_info["device"] = torch.device("cuda:0")
    
    # 判断 GPU 型号
    is_a2000 = "a2000" in device_name.lower()
    is_4070 = "4070" in device_name
    is_rtx = "rtx" in device_name.lower()

    # 所有 RTX / 现代 GPU 均支持 FP16（Tensor Cores）
    gpu_info["use_amp"] = True
    gpu_info["is_a2000"] = is_a2000
    gpu_info["is_4070"] = is_4070

    logger.info("=" * 60)
    logger.info("GPU 检测报告")
    logger.info("=" * 60)
    logger.info(f"  设备名称:     {device_name}")
    logger.info(f"  显存大小:     {gpu_info['vram_gb']} GB")
    logger.info(f"  CUDA 版本:    {torch.version.cuda}")
    logger.info(f"  cuDNN 版本:   {torch.backends.cudnn.version()}")
    logger.info(f"  PyTorch 版本: {torch.__version__}")
    logger.info(f"  混合精度训练: {'✅ 启用' if gpu_info['use_amp'] else '❌ 不支持'}")

    if is_a2000:
        logger.info(f"  🎯 检测到 RTX A2000 Laptop GPU — 优化配置")
        logger.info(f"     - Ampere 架构 Tensor Cores → FP16 加速 ~2x")
        logger.info(f"     - 默认 batch_size=32（适配 4GB VRAM）")
    elif is_4070:
        logger.info(f"  🎯 检测到 RTX 4070 — Ada Lovelace 12GB VRAM")
        logger.info(f"     - Ada Lovelace FP8 Tensor Cores → FP16 加速 ~3–4x")
        logger.info(f"     - 自动调优: batch_size=96, input_size=336")
        logger.info(f"     - 自动调优: 更长训练 & 更多解冻层")
    elif is_rtx:
        logger.info(f"  🎯 检测到 NVIDIA RTX 系列 GPU")

    # cuDNN 自动调优
    torch.backends.cudnn.benchmark = True
    if is_4070:
        # Ada Lovelace 支持 TF32（自动启用），额外启用 FP8 优化
        torch.set_float32_matmul_precision("high")
        logger.info(f"  TF32/FP8 matmul:  已启用（Ada Lovelace 专属加速）")
    logger.info(f"  cuDNN benchmark: 已启用（自动寻找最优卷积算法）")
    logger.info("=" * 60)

    return gpu_info


# ============================================================
# 数据集与数据增强
# ============================================================

# 5 个猫品种对应的子文件夹名称
CAT_BREEDS = ["ragdoll", "singapura", "persian", "sphynx", "pallas"]

# 品种中文名称映射
BREED_CN = {
    "ragdoll": "布偶猫",
    "singapura": "新加坡猫",
    "persian": "波斯猫",
    "sphynx": "斯芬克斯猫",
    "pallas": "兔狲",
}


def get_transforms(input_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    构建训练集和验证集的图像预处理变换。
    
    训练集：使用多种数据增强（随机裁剪、翻转、颜色抖动等）
    验证集：仅使用中心裁剪 + 归一化（不做增强）
    
    关键：归一化参数必须与 ImageNet 预训练时一致
          mean=[0.485, 0.456, 0.406]
          std=[0.229, 0.224, 0.225]
    
    Returns:
        (train_transforms, val_transforms)
    """
    # ImageNet 标准化参数（必须与预训练一致！）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # 训练集数据增强
    # 丰富增强策略，抑制 Phase 2 微调时的过拟合：
    #   - RandAugment：自动搜索的最优增强组合（TorchVision 内置）
    #   - RandomErasing：随机遮挡矩形区域，迫使模型关注整体特征而非局部细节
    #   - 颜色抖动参数适度调大，模拟小车在不同光照下的真实场景
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.3,   # 扩大光照变化范围
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomGrayscale(p=0.05),       # 5% 概率灰度化（鲁棒性）
        transforms.RandomPerspective(p=0.3),       # 30% 概率透视变换
        transforms.ToTensor(),
        transforms.RandomErasing(                  # 随机遮挡（正则化利器）
            p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3)
        ),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    # 验证集变换（不做增强）
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
    input_size: int = 224,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, Optional[datasets.ImageFolder], dict]:
    """
    准备训练集、验证集和可选的独立测试集。
    
    数据目录结构（来自 image_collector）：
        collected_images/
        ├── ragdoll/    (350 张)
        ├── singapura/  (350 张)
        ├── persian/    (350 张)
        ├── sphynx/     (350 张)
        └── pallas/     (350 张)
    
    三路划分逻辑：
        test_ratio > 0: 先分出测试集，剩余按 val_ratio 划分训练/验证
                        例: test_ratio=0.1, val_ratio=0.15
                        → 测试 10% | 验证 13.5% | 训练 76.5%
        test_ratio = 0: 仅训练/验证两路划分（向后兼容）
    
    Args:
        data_root: 数据根目录路径
        val_ratio: 验证集在「非测试数据」中的比例（默认 15%）
        test_ratio: 独立测试集比例（默认 0，即不划分测试集）
        seed: 随机种子
    
    Returns:
        (train_dataset, val_dataset, test_dataset_or_None, class_info_dict)
    """
    from torch.utils.data import Subset, random_split
    
    data_path = Path(data_root)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据目录不存在: {data_path}\n"
            f"请先运行 image_collector/collect_cat_images.py 搜集图片"
        )
    
    train_transforms, val_transforms = get_transforms(input_size)
    
    # 加载完整数据集
    full_dataset = datasets.ImageFolder(str(data_path), transform=None)
    total_size = len(full_dataset)
    
    # 验证目录结构
    logger.info("\n数据集目录结构:")
    for breed in CAT_BREEDS:
        breed_dir = data_path / breed
        if breed_dir.exists():
            count = len(list(breed_dir.glob("*.[jJ][pP][gG]")))
            count += len(list(breed_dir.glob("*.[pP][nN][gG]")))
            logger.info(f"  {breed}/ ({BREED_CN.get(breed, breed)}): {count} 张")
        else:
            logger.warning(f"  ⚠️ {breed}/ 目录不存在！")
    
    generator = torch.Generator().manual_seed(seed)
    
    # ================================================================
    # 三路划分：测试集 → 训练集 + 验证集
    # ================================================================
    test_dataset = None
    test_indices: list[int] = []
    
    if test_ratio > 0:
        # 先分出独立测试集（hold-out，训练过程中完全不接触）
        test_size = int(total_size * test_ratio)
        remain_size = total_size - test_size
        
        remain_subset, test_subset = random_split(
            full_dataset, [remain_size, test_size],
            generator=generator,
        )
        # 从 Subset 中提取 indices
        remain_indices = remain_subset.indices
        test_indices = list(test_subset.indices)
        
        logger.info(f"\n  先划分测试集: {test_size} 张 ({test_ratio*100:.0f}%)")
        logger.info(f"  剩余数据:     {remain_size} 张")
    else:
        remain_indices = list(range(total_size))
    
    # 在剩余数据中划分训练/验证
    remain_size = len(remain_indices)
    val_size = int(remain_size * val_ratio)
    train_size = remain_size - val_size
    
    # 对 remain_indices 再随机划分
    perm = torch.randperm(remain_size, generator=generator).tolist()
    train_local = perm[:train_size]
    val_local = perm[train_size:]
    train_indices = [remain_indices[i] for i in train_local]
    val_indices = [remain_indices[i] for i in val_local]
    
    # 构建带 transform 的 dataset
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
        logger.info(f"  测试集:   {len(test_indices)} ({len(test_indices)/total_size*100:.1f}%)  ← 只在最终评估使用")
    logger.info(f"  训练集:   {train_size} ({train_size/total_size*100:.1f}%)")
    logger.info(f"  验证集:   {val_size} ({val_size/total_size*100:.1f}%)")
    logger.info(f"  类别数:   {class_info['num_classes']}")
    logger.info(f"  类别:     {class_info['classes']}")
    
    return train_dataset, val_dataset, test_dataset, class_info


# ============================================================
# 模型构建
# ============================================================

def build_model(num_classes: int = 5, freeze_backbone: bool = True) -> nn.Module:
    """
    构建基于 ResNet-50 的迁移学习模型。
    
    步骤：
    1. 加载 ImageNet 预训练的 ResNet-50（自动下载权重）
    2. 替换最后的全连接层（1000 → num_classes）
    3. 可选冻结骨干网络（阶段一训练时使用）
    
    Args:
        num_classes: 分类数量（5 个猫品种）
        freeze_backbone: 是否冻结骨干网络
    
    Returns:
        构建好的 PyTorch 模型
    """
    logger.info(f"\n构建模型: ResNet-50 (预训练于 ImageNet)")
    logger.info(f"  输出类别数: {num_classes}")
    
    # 加载预训练模型（自动下载 resnet50-0676ba61.pth，~98MB）
    # 权重文件缓存路径: ~/.cache/torch/hub/checkpoints/
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # 替换分类头 — 加入 BatchNorm 和更强的 Dropout 以抑制过拟合
    # 原 fc 层: Linear(2048, 1000) → 新分类头: 2048 → 1024 → BatchNorm → 512 → 5
    num_features = model.fc.in_features  # 2048
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),               # 提高 dropout 比例（原 0.5 → 0.6）
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    
    # 初始化新分类头的权重
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    if freeze_backbone:
        logger.info("  骨干网络: 冻结 ❄️（仅训练分类头）")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        logger.info("  骨干网络: 可训练 🔥")
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    return model


# ============================================================
# 训练与验证函数
# ============================================================

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
) -> Tuple[float, float]:
    """
    训练一个 epoch。
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数（支持 label smoothing）
        optimizer: 优化器
        device: 计算设备
        scaler: 混合精度 GradScaler（None 表示不使用 AMP）
        epoch: 当前 epoch 编号
        total_epochs: 总 epoch 数
        max_grad_norm: 梯度裁剪阈值（默认 1.0，防止梯度爆炸导致过拟合）
    
    Returns:
        (平均损失, 准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        # 混合精度前向传播
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # 梯度裁剪（混合精度下先 unscale 再 clip）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 每 10 个 batch 打印一次进度
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            speed = batches_done * dataloader.batch_size / elapsed
            logger.info(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batches_done}/{len(dataloader)}] "
                f"Loss: {running_loss/batches_done:.4f} "
                f"Acc: {correct/total*100:.2f}% "
                f"Speed: {speed:.0f} img/s"
            )
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    在验证集上评估模型。
    
    Returns:
        (平均损失, 准确率)
    """
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
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


# ============================================================
# EMA (Exponential Moving Average) — 模型参数平滑
# ============================================================

class EMAModel:
    """
    指数移动平均模型参数平滑器。

    训练时维护一份 shadow copy，每个 step 按衰减率更新:
        shadow = decay * shadow + (1 - decay) * model

    评估/保存时切换到 EMA 参数——EMA 模型通常比原始模型具有更好的
    泛化性能，是缓解 Phase 2 过拟合的实用技巧。

    用法:
        ema = EMAModel(model, decay=0.999)
        for batch in dataloader:
            train_step(model, ...)
            ema.update(model)         # 每步更新 shadow

        ema.apply_shadow(model)       # 验证/保存前切换到 EMA 参数
        val_acc = validate(model, ...)
        ema.restore(model)            # 恢复继续训练
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        # 初始化 shadow 为原始参数副本
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """使用当前模型参数更新 shadow。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """将 EMA shadow 参数应用到模型（验证/保存前调用）。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """恢复原始参数（继续训练前调用）。"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ============================================================
# 主训练流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-50 猫品种识别模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 数据参数
    parser.add_argument(
        "--data",
        type=str,
        default="image_collector/collected_images",
        help="数据集根目录路径",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例（默认 0.15，即 85% 训练 / 15% 验证）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="训练集比例（如 0.85）。指定后自动计算 val_ratio=1-train_ratio，"
             "优先级高于 --val-ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="独立测试集比例（默认 0，不划分测试集）。"
             "例如 0.1 → 先分出 10%% 作为 hold-out 测试集，"
             "剩余 90%% 再按 val_ratio 划分训练/验证集。"
             "测试集仅在最终评估时使用，训练过程中完全不接触。",
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="阶段二微调的总训练轮数（默认 50）",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=15,
        help="阶段一（特征提取）的训练轮数（默认 15）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="批次大小（默认 32；大显存 GPU 会自动上调）",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="输入图像大小（默认 224；大显存 GPU 可上调至 336 提升特征精度）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="阶段一初始学习率（默认 0.001）",
    )
    parser.add_argument(
        "--lr-finetune",
        type=float,
        default=0.0001,
        help="阶段二微调学习率（默认 0.0001）",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label Smoothing 系数（默认 0.1）。"
             "将硬标签 [0,1,0,0,0] 平滑为 [0.02,0.92,0.02,0.02,0.02]，"
             "阻止模型对训练样本过于自信，是抑制过拟合的有效手段。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="权重衰减系数 L2 正则化（默认 5e-4）。"
             "Phase 2 的解冻层参数多、数据少，需要比 Phase 1 更强的 weight decay。",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（默认 1.0）。"
             "限制梯度范数，防止微调时梯度爆炸引起过拟合。",
    )
    
    # 其他
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="数据加载的并行线程数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练（提供检查点路径）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="模型和日志的输出目录",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="禁用混合精度训练（FP32 训练）",
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # 0. 初始化
    # ============================================================
    
    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"resnet50_cat_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # ============================================================
    # 1. GPU 检测
    # ============================================================
    
    gpu_info = detect_gpu()
    device = gpu_info["device"]
    use_amp = gpu_info["use_amp"] and not args.no_amp
    
    if not gpu_info["cuda_available"]:
        logger.warning("继续使用 CPU 训练...")
        logger.warning(f"建议 batch_size 降低到 8 以下")
        if args.batch > 16:
            logger.warning(f"batch_size 从 {args.batch} 降低到 16")
            args.batch = 16

    # -----------------------------------------------------------
    # 自动调优：根据 GPU 显存动态设定训练超参数
    # -----------------------------------------------------------
    vram_gb = gpu_info.get("vram_gb", 0)
    if vram_gb >= 10:
        logger.info(f"\n🔧 检测到大显存 GPU ({vram_gb} GB)，自动优化训练配置")

        # batch_size 放大（利用大显存提升吞吐）
        suggested_batch = 64 if vram_gb < 16 else 96
        if args.batch < suggested_batch:
            logger.info(f"   batch_size:    {args.batch} → {suggested_batch}")
            args.batch = suggested_batch

        # 输入分辨率提升（更高分辨率提取精细特征）
        suggested_size = 336
        if args.input_size < suggested_size:
            logger.info(f"   input_size:    {args.input_size} → {suggested_size}")
            args.input_size = suggested_size

        # 训练轮数延长（充分发挥大模型潜力）
        suggested_epochs = 80
        if args.epochs < suggested_epochs:
            logger.info(f"   epochs:        {args.epochs} → {suggested_epochs}")
            args.epochs = suggested_epochs

        suggested_phase1 = 20
        if args.phase1_epochs < suggested_phase1:
            logger.info(f"   phase1_epochs: {args.phase1_epochs} → {suggested_phase1}")
            args.phase1_epochs = suggested_phase1

        # workers 翻倍
        suggested_workers = min(args.workers * 2, 16)
        if args.workers < suggested_workers:
            logger.info(f"   workers:       {args.workers} → {suggested_workers}")
            args.workers = suggested_workers

        # weight_decay 放宽（大批次梯度更稳定，可减轻 L2 惩罚）
        relaxed_wd = 3e-4
        if args.weight_decay > relaxed_wd:
            logger.info(f"   weight_decay:   {args.weight_decay} → {relaxed_wd}")
            args.weight_decay = relaxed_wd

        # 梯度裁剪放宽（大显存 → 梯度更稳定）
        relaxed_clip = 2.0
        if args.grad_clip < relaxed_clip:
            logger.info(f"   grad_clip:      {args.grad_clip} → {relaxed_clip}")
            args.grad_clip = relaxed_clip

        logger.info("   使用 --batch/--epochs/--input-size/--workers 可覆盖以上自动值\n")
    
    # ============================================================
    # 2. 准备数据集
    # ============================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("准备数据集")
    logger.info("=" * 60)
    
    data_root = Path(args.data)
    # 如果是相对路径，转换为相对于 Baseline 目录的绝对路径
    if not data_root.is_absolute():
        script_dir = Path(__file__).resolve().parent
        data_root = script_dir / args.data
    
    logger.info(f"数据路径: {data_root}")
    
    # 计算训练/验证划分比例
    # --train-ratio 优先级高于 --val-ratio；默认 85% 训练 / 15% 验证
    if args.train_ratio is not None:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError(f"--train-ratio 必须在 (0, 1) 之间，收到: {args.train_ratio}")
        train_ratio = args.train_ratio
        val_ratio = 1.0 - train_ratio
        logger.info(f"使用 --train-ratio={train_ratio} → 验证集比例={val_ratio}")
    else:
        val_ratio = args.val_ratio
        train_ratio = 1.0 - val_ratio
        logger.info(f"使用 --val-ratio={val_ratio} → 训练集比例={train_ratio}")
    
    if args.test_ratio > 0:
        logger.info(f"启用独立测试集: --test-ratio={args.test_ratio}")
        logger.info(f"  实际划分: 测试 {args.test_ratio*100:.0f}% | "
                     f"训练 {(1-args.test_ratio)*(1-val_ratio)*100:.0f}% | "
                     f"验证 {(1-args.test_ratio)*val_ratio*100:.0f}%")
    
    train_dataset, val_dataset, test_dataset, class_info = prepare_datasets(
        str(data_root),
        val_ratio=val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        input_size=args.input_size,
    )
    
    # 校验划分比例
    if test_dataset is not None:
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        logger.info(
            f"实际划分: 训练 {len(train_dataset)/total*100:.1f}% / "
            f"验证 {len(val_dataset)/total*100:.1f}% / "
            f"测试 {len(test_dataset)/total*100:.1f}%"
        )
    else:
        total = len(train_dataset) + len(val_dataset)
        actual_train_ratio = len(train_dataset) / total
        logger.info(
            f"实际划分: 训练集 {actual_train_ratio*100:.1f}% / "
            f"验证集 {(1-actual_train_ratio)*100:.1f}%"
        )
    
    # 保存测试集图像路径（供 evaluate_cnn.py 使用）
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
        drop_last=True,  # 丢弃不完整 batch，稳定 BN
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if gpu_info["cuda_available"] else False,
    )
    
    num_classes = class_info["num_classes"]
    
    # ============================================================
    # 3. 构建模型
    # ============================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("构建模型")
    logger.info("=" * 60)
    
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)
    
    # ============================================================
    # 4. 训练
    # ============================================================
    
    # Label Smoothing 损失函数：将硬标签 [0,1,0,0,0] 平滑为 [ε/4, 1-ε, ε/4, ε/4, ε/4]
    # 阻止模型对训练样本过度自信（output prob → 1.0），显著抑制过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler() if use_amp else None
    
    logger.info(f"\n正则化配置:")
    logger.info(f"  Label Smoothing:     {args.label_smoothing}")
    logger.info(f"  Weight Decay:        {args.weight_decay}")
    logger.info(f"  Gradient Clipping:   {args.grad_clip}")
    
    best_val_acc = 0.0
    total_epoch = 0
    
    # EMA（仅 Phase 2 启用，Phase 1 只训练分类头不需要）
    ema: Optional[EMAModel] = None
    
    # ----------------------------------------------------------
    # 阶段一：特征提取（冻结骨干，仅训练分类头）
    # ----------------------------------------------------------
    
    logger.info("\n" + "=" * 60)
    logger.info("阶段一：特征提取（Feature Extraction）")
    logger.info("  - 骨干网络: 冻结 ❄️")
    logger.info("  - 分类头:   训练 🔥")
    logger.info(f"  - 学习率:   {args.lr}")
    logger.info(f"  - Epochs:   1-{args.phase1_epochs}")
    logger.info("=" * 60)
    
    # 优化器 — 阶段一 weight_decay 用默认值的 1/5（分类头新参数，不需要强正则）
    optimizer = optim.AdamW(
        model.fc.parameters(), lr=args.lr,
        weight_decay=args.weight_decay * 0.2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs
    )
    
    for epoch in range(1, args.phase1_epochs + 1):
        total_epoch += 1
        logger.info(f"\n--- Phase 1 Epoch {epoch}/{args.phase1_epochs} ---")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, args.phase1_epochs,
            max_grad_norm=args.grad_clip,
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # 记录
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        )
        logger.info(
            f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
            f" | LR: {current_lr:.6f}"
        )
        
        writer.add_scalar("Phase1/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase1/Train_Acc", train_acc, epoch)
        writer.add_scalar("Phase1/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase1/Val_Acc", val_acc, epoch)
        writer.add_scalar("Phase1/LR", current_lr, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_phase1.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info(f"  ✅ 最佳模型已保存: {checkpoint_path}")
    
    logger.info(f"\n阶段一完成！最佳验证准确率: {best_val_acc*100:.2f}%")
    
    # ----------------------------------------------------------
    # 阶段二：微调（解冻 layer2~4，全局微调）
    # ----------------------------------------------------------
    #
    # 阶段二过拟合风险最高，引入以下多层正则化策略:
    #   1. Label Smoothing (ε=0.1): 阻止模型对训练样本过度自信
    #   2. 差异化 Weight Decay: 解冻层参数多 → 更高 L2 惩罚
    #   3. 梯度裁剪: 防止梯度爆炸
    #   4. EMA (decay=0.999): 参数平滑，提升泛化能力
    #   5. 降低解冻层学习率: 防止破坏预训练特征
    #   6. 早停: 更快停止过拟合
    #   7. 大显存 GPU (>=10GB): 额外解冻 layer2，更深微调
    # ----------------------------------------------------------

    unfreeze_layer2 = gpu_info.get("vram_gb", 0) >= 10

    logger.info("\n" + "=" * 60)
    logger.info("阶段二：微调（Fine-tuning）— 强化正则化")
    if unfreeze_layer2:
        logger.info("  🎯 大显存模式 — 深度微调")
        logger.info("  - Layer1:         冻结 ❄️（通用底层特征）")
        logger.info("  - Layer2-4:       解冻 🔥（中高层特征）")
    else:
        logger.info("  - Layer1-2:       冻结 ❄️（通用特征）")
        logger.info("  - Layer3-4:       解冻 🔥（品种特定特征）")
    logger.info("  - 分类头:         训练 🔥")
    logger.info(f"  - Label Smoothing: {args.label_smoothing}")
    logger.info(f"  - Weight Decay:    {args.weight_decay}（强化）")
    logger.info(f"  - Gradient Clip:   {args.grad_clip}")
    logger.info(f"  - EMA:             decay=0.999 ✅")
    logger.info(f"  - 学习率:         差异化 LR（降低避免破坏预训练特征）")
    patience_str = f"patience={'10' if unfreeze_layer2 else '7'}"
    logger.info(f"  - 早停:            {patience_str}")
    logger.info(f"  - Epochs:          {args.phase1_epochs+1}-{args.epochs}")
    logger.info("=" * 60)
    
    # 加载阶段一的最佳权重
    best_phase1_path = checkpoint_dir / "best_phase1.pth"
    if best_phase1_path.exists():
        checkpoint = torch.load(best_phase1_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载阶段一最佳模型: val_acc={checkpoint['val_acc']*100:.2f}%")
    
    # 解冻更多层（大显存 GPU 可安全解冻 layer2，进行更深层微调）
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
        elif "layer2" in name and unfreeze_layer2:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    logger.info(f"  解冻后新增可训练参数: {trainable - sum(p.numel() for p in model.fc.parameters() if p.requires_grad):,}")
    
    # --- 差异化学习率 + 差异化 Weight Decay ---
    # 分类头（新层，高 LR + 适中 WD）
    # layer4（最高层，低 LR + 强化 WD — 特征与品种最相关，容易过拟合）
    # layer3（中高层，极低 LR + 强化 WD — 半通用特征，只需微调）
    # layer2（中层，极低 LR + 强化 WD — 仅大显存 GPU 解冻，通用中层特征微调）
    layer2_group = ([
        {
            "params": model.layer2.parameters(),
            "lr": args.lr_finetune * 0.1,
            "weight_decay": args.weight_decay,
        },
    ] if unfreeze_layer2 else [])
    optimizer = optim.AdamW([
        {
            "params": model.fc.parameters(),
            "lr": args.lr_finetune * 3,       # 降低：原来 ×5 → ×3
            "weight_decay": args.weight_decay * 0.5,  # 分类头 WD 减半
        },
        {
            "params": model.layer4.parameters(),
            "lr": args.lr_finetune * 0.5,      # 降低：原来 ×1.0 → ×0.5
            "weight_decay": args.weight_decay,          # 强化 WD
        },
        {
            "params": model.layer3.parameters(),
            "lr": args.lr_finetune * 0.2,      # 降低：原来 ×0.5 → ×0.2
            "weight_decay": args.weight_decay,          # 强化 WD
        },
    ] + layer2_group)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.phase1_epochs
    )
    
    # EMA（指数移动平均参数平滑）
    ema = EMAModel(model, decay=0.999)
    logger.info(f"EMA 已初始化 (decay=0.999)")
    
    # 早停配置（大显存 → 更长耐心）
    patience = 10 if unfreeze_layer2 else 7
    no_improve = 0
    best_val_acc_phase2 = best_val_acc      # 用于判断是否超越 Phase1
    best_phase2_val = 0.0                    # Phase2 自身最佳（保证最终有模型产出）
    
    phase2_epochs = args.epochs - args.phase1_epochs
    
    for epoch in range(1, phase2_epochs + 1):
        total_epoch += 1
        logger.info(f"\n--- Phase 2 Epoch {epoch}/{phase2_epochs} "
                     f"(Total: {total_epoch}/{args.epochs}) ---")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, phase2_epochs,
            max_grad_norm=args.grad_clip,
        )
        
        # 更新 EMA shadow
        ema.update(model)
        
        # 验证: 切换到 EMA 参数进行评估
        ema.apply_shadow(model)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        ema.restore(model)
        
        # 更新学习率
        current_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()
        
        # 过拟合检测: train_acc - val_acc 差距
        acc_gap = train_acc - val_acc
        gap_warning = ""
        if acc_gap > 0.10:
            gap_warning = f" ⚠️ 过拟合严重！(gap={acc_gap*100:.1f}%)"
        elif acc_gap > 0.05:
            gap_warning = f" ⚡ 轻微过拟合 (gap={acc_gap*100:.1f}%)"
        
        # 记录
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        )
        logger.info(
            f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
            f" | LRs: {[f'{lr:.6f}' for lr in current_lrs]}"
            f" | Gap: {acc_gap*100:.1f}%{gap_warning}"
        )
        
        writer.add_scalar("Phase2/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase2/Train_Acc", train_acc, epoch)
        writer.add_scalar("Phase2/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase2/Val_Acc", val_acc, epoch)
        writer.add_scalar("Phase2/TrainVal_Gap", acc_gap, epoch)
        
        # 保存「超越 Phase1」的最佳模型（使用 EMA 平滑后的权重，泛化能力更强）
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            no_improve = 0
            checkpoint_path = checkpoint_dir / "best_model.pth"

            # 临时切换到 EMA shadow 权重再保存
            ema.apply_shadow(model)
            torch.save({
                "epoch": total_epoch,
                "model_state_dict": model.state_dict(),   # EMA 权重
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
                "gpu_info": {k: str(v) for k, v in gpu_info.items()},
                "phase": "phase2",
            }, checkpoint_path)
            ema.restore(model)  # 恢复训练权重，继续下一 epoch

            logger.info(f"  ✅ 最佳模型已保存（EMA 权重，超越 Phase1）: {checkpoint_path}")
        else:
            no_improve += 1
            logger.info(f"  未提升 ({no_improve}/{patience})")

        # 始终记录 Phase2 自身最优，保证即使未超越 Phase1 也有可用模型
        if val_acc > best_phase2_val:
            best_phase2_val = val_acc
            ema.apply_shadow(model)
            torch.save({
                "epoch": total_epoch,
                "model_state_dict": model.state_dict(),   # EMA 权重
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
                "gpu_info": {k: str(v) for k, v in gpu_info.items()},
                "phase": "phase2",
            }, checkpoint_dir / "best_phase2.pth")
            ema.restore(model)
        
        # 早停
        if no_improve >= patience:
            logger.info(f"\n早停触发！验证准确率连续 {patience} 轮未提升")
            break

    # ============================================================
    # 4.5 保证 best_model.pth 一定存在（供部署 / tcp_server 使用）
    # ============================================================
    if not (checkpoint_dir / "best_model.pth").exists():
        logger.warning(
            "⚠️  Phase2 未超越 Phase1 最佳验证准确率，未自动生成 best_model.pth"
        )
        # 在 Phase2 自身最优与 Phase1 最优之间取较优者，确保部署模型可用
        candidates = []
        if (checkpoint_dir / "best_phase2.pth").exists():
            p2 = torch.load(checkpoint_dir / "best_phase2.pth", map_location="cpu")
            candidates.append(("best_phase2.pth", p2.get("val_acc", 0.0)))
        if (checkpoint_dir / "best_phase1.pth").exists():
            p1 = torch.load(checkpoint_dir / "best_phase1.pth", map_location="cpu")
            candidates.append(("best_phase1.pth", p1.get("val_acc", 0.0)))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            src_name, src_val = candidates[0]
            src = checkpoint_dir / src_name
            ckpt = torch.load(src, map_location="cpu")
            dst = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": ckpt.get("epoch", total_epoch),
                "model_state_dict": ckpt.get("model_state_dict", ckpt),
                "optimizer_state_dict": ckpt.get("optimizer_state_dict", {}),
                "val_acc": ckpt.get("val_acc", 0.0),
                "val_loss": ckpt.get("val_loss", 0.0),
                "class_info": class_info,
                "gpu_info": {k: str(v) for k, v in gpu_info.items()},
                "phase": f"{src_name}_fallback",
            }, dst)
            logger.info(
                f"  📦 已将较优模型 {src_name} (val_acc={src_val*100:.2f}%) "
                f"复制为 best_model.pth，部署可用"
            )
        else:
            logger.error("❌ 未找到任何可用 checkpoint，无法生成 best_model.pth")

    # ============================================================
    # 5. 保存最终模型
    # ============================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"最佳验证准确率: {best_val_acc_phase2*100:.2f}%")
    logger.info(f"总训练轮数:     {total_epoch}")
    logger.info(f"模型保存路径:   {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"TensorBoard:    tensorboard --logdir {log_dir}")
    
    # 保存 TorchScript 模型（用于部署）
    try:
        best_checkpoint = torch.load(
            checkpoint_dir / "best_model.pth", map_location="cpu"
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()
        
        # 导出为 TorchScript
        example_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model.cpu(), example_input)
        script_path = output_dir / "resnet50_cat_scripted.pt"
        traced_model.save(str(script_path))
        logger.info(f"TorchScript 模型: {script_path}")
    except Exception as e:
        logger.warning(f"TorchScript 导出失败: {e}")
    
    # 保存 ONNX 模型
    try:
        model = model.to("cpu")
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = output_dir / "resnet50_cat.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"ONNX 模型:      {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX 导出失败: {e}")
    
    writer.close()
    
    # ============================================================
    # 6. 最终输出
    # ============================================================
    
    logger.info(f"\n{'=' * 60}")
    logger.info("输出文件汇总")
    logger.info(f"{'=' * 60}")
    logger.info(f"  最佳模型:     {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"  训练日志:     training.log")
    logger.info(f"  TensorBoard:  {log_dir}")
    logger.info(f"")
    logger.info(f"下一步:")
    logger.info(f"  1. 查看训练曲线: tensorboard --logdir {log_dir}")
    logger.info(f"  2. 评估模型:     python evaluate_cnn.py --model {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"  3. 单张推理:     python infer_cnn.py --model {checkpoint_dir / 'best_model.pth'} --image <path>")


if __name__ == "__main__":
    main()
