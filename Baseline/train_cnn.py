"""
猫品种识别模型 — ResNet-50 迁移学习训练脚本
==============================================

基于 baseline_prompt.md 中的分析：
  - 推荐 CNN 架构（ResNet-50）而非 YOLO，因任务本质是 5 猫品种细粒度分类
  - 使用 ImageNet 预训练权重进行两阶段迁移学习
  - 针对 NVIDIA RTX A2000 Laptop GPU (4GB GDDR6) 优化

两阶段训练策略：
  阶段一（特征提取）：冻结骨干网络，仅训练新分类头
  阶段二（微调）：     解冻 layer3-4，差异化学习率全局微调

GPU 支持：
  自动检测 RTX A2000 并启用 CUDA + FP16 混合精度训练
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
    
    针对 NVIDIA RTX A2000 Laptop GPU 做特别检测和报告。
    
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
    
    # 判断是否为 RTX A2000
    is_a2000 = "a2000" in device_name.lower()
    is_rtx = "rtx" in device_name.lower()
    
    # RTX A2000 支持 FP16 混合精度（Ampere 架构 Tensor Cores）
    gpu_info["use_amp"] = True
    
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
        logger.info(f"  🎯 检测到 RTX A2000 Laptop GPU — 使用优化配置")
        logger.info(f"     - Ampere 架构 Tensor Cores → FP16 加速 ~2x")
        logger.info(f"     - 推荐 batch_size=32（充分利用 4GB VRAM）")
    elif is_rtx:
        logger.info(f"  🎯 检测到 NVIDIA RTX 系列 GPU")
    
    # cuDNN 自动调优
    torch.backends.cudnn.benchmark = True
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


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
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
    # 模拟小车在实际环境中可能遇到的光照、角度变化
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,   # 模拟不同光照条件
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    # 验证集变换（不做增强）
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    return train_transforms, val_transforms


def prepare_datasets(
    data_root: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    seed: int = 42,
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
    
    train_transforms, val_transforms = get_transforms()
    
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
    
    # 替换分类头
    # 原 fc 层: Linear(2048, 1000) → 新分类头: 2048 → 512 → 5
    num_features = model.fc.in_features  # 2048
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
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
) -> Tuple[float, float]:
    """
    训练一个 epoch。
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        scaler: 混合精度 GradScaler（None 表示不使用 AMP）
        epoch: 当前 epoch 编号
        total_epochs: 总 epoch 数
    
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
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
        help="批次大小（默认 32，适配 RTX A2000 4GB VRAM）",
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
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None
    
    best_val_acc = 0.0
    total_epoch = 0
    
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
    
    # 优化器（只优化分类头参数）
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
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
    # 阶段二：微调（解冻 layer3-4，全局微调）
    # ----------------------------------------------------------
    
    logger.info("\n" + "=" * 60)
    logger.info("阶段二：微调（Fine-tuning）")
    logger.info("  - Layer1-2: 冻结 ❄️（通用特征）")
    logger.info("  - Layer3-4: 解冻 🔥（品种特定特征）")
    logger.info("  - 分类头:   训练 🔥")
    logger.info(f"  - 学习率:   差异化 LR")
    logger.info(f"  - Epochs:   {args.phase1_epochs+1}-{args.epochs}")
    logger.info("=" * 60)
    
    # 加载阶段一的最佳权重
    best_phase1_path = checkpoint_dir / "best_phase1.pth"
    if best_phase1_path.exists():
        checkpoint = torch.load(best_phase1_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载阶段一最佳模型: val_acc={checkpoint['val_acc']*100:.2f}%")
    
    # 解冻 layer3 和 layer4
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    # 差异化学习率
    # 分类头（新层）: 较高 LR
    # layer4（最高层）: 中等 LR
    # layer3（中高层）: 较低 LR
    optimizer = optim.AdamW([
        {"params": model.fc.parameters(),       "lr": args.lr_finetune * 5},
        {"params": model.layer4.parameters(),   "lr": args.lr_finetune},
        {"params": model.layer3.parameters(),   "lr": args.lr_finetune * 0.5},
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.phase1_epochs
    )
    
    # 早停配置
    patience = 10
    no_improve = 0
    best_val_acc_phase2 = best_val_acc
    
    phase2_epochs = args.epochs - args.phase1_epochs
    
    for epoch in range(1, phase2_epochs + 1):
        total_epoch += 1
        logger.info(f"\n--- Phase 2 Epoch {epoch}/{phase2_epochs} "
                     f"(Total: {total_epoch}/{args.epochs}) ---")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, phase2_epochs,
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        current_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()
        
        # 记录
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        )
        logger.info(
            f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%"
            f" | LRs: {[f'{lr:.6f}' for lr in current_lrs]}"
        )
        
        writer.add_scalar("Phase2/Train_Loss", train_loss, epoch)
        writer.add_scalar("Phase2/Train_Acc", train_acc, epoch)
        writer.add_scalar("Phase2/Val_Loss", val_loss, epoch)
        writer.add_scalar("Phase2/Val_Acc", val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            no_improve = 0
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": total_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_info": class_info,
                "gpu_info": {k: str(v) for k, v in gpu_info.items()},
            }, checkpoint_path)
            logger.info(f"  ✅ 最佳模型已保存: {checkpoint_path}")
        else:
            no_improve += 1
            logger.info(f"  未提升 ({no_improve}/{patience})")
        
        # 早停
        if no_improve >= patience:
            logger.info(f"\n早停触发！验证准确率连续 {patience} 轮未提升")
            break
    
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
