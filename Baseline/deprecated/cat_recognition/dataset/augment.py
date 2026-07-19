"""教室场景特定的数据增强策略。

小车摄像头从低角度拍摄，猫可能：
1. 出现在地面、桌面等位置
2. 部分被桌椅遮挡
3. 光线变化大（教室灯光、窗户自然光）
4. 距离变化大（远处小目标 → 近处大目标）

数据增强模拟这些场景，提升模型鲁棒性。
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def create_classroom_augmentation() -> A.Compose:
    """创建教室场景专用的数据增强流水线。

    针对低角度、遮挡、光线变化等教室场景特点。
    """
    return A.Compose(
        [
            # ---- 几何变换（模拟不同拍摄角度）----
            A.RandomResizedCrop(
                height=640,
                width=640,
                scale=(0.5, 1.0),      # 模拟不同距离
                ratio=(0.75, 1.33),     # 模拟不同宽高比
                p=0.5,
            ),
            A.Affine(
                scale=(0.8, 1.2),       # 模拟远近变化
                translate_percent=(-0.1, 0.1),  # 模拟目标偏移
                rotate=(-15, 15),        # 模拟摄像头倾斜
                shear=(-5, 5),           # 模拟视角扭曲
                p=0.5,
            ),
            A.Perspective(
                scale=(0.02, 0.08),     # 模拟低角度透视
                p=0.3,
            ),

            # ---- 遮挡模拟（教室桌椅遮挡）----
            A.CoarseDropout(
                max_holes=3,
                max_height=80,
                max_width=80,
                min_holes=1,
                min_height=20,
                min_width=20,
                fill_value=0,
                p=0.3,
            ),

            # ---- 光线变化模拟 ----
            A.RandomBrightnessContrast(
                brightness_limit=0.2,    # 模拟灯光变化
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,      # 色调变化
                sat_shift_limit=30,      # 饱和度变化
                val_shift_limit=30,      # 亮度变化
                p=0.3,
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),   # 伽马校正
                p=0.2,
            ),

            # ---- 模糊与噪声（模拟摄像头运动模糊）----
            A.MotionBlur(
                blur_limit=(3, 7),
                p=0.2,
            ),
            A.GaussNoise(
                var_limit=(10.0, 30.0),  # 传感器噪声
                p=0.2,
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.3),    # 低光照高 ISO 噪声
                p=0.15,
            ),

            # ---- 压缩伪影（模拟 JPEG 传输）----
            A.ImageCompression(
                quality_lower=60,        # 模拟 quality=60-70 的 JPEG 压缩
                quality_upper=85,
                p=0.3,
            ),

            # ---- 标准化 ----
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",              # YOLO 格式 [x_center, y_center, w, h] 归一化
            label_fields=["class_labels"],
            min_visibility=0.3,         # 增强后标注框可见度至少 30%
        ),
    )


def apply_augmentation(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int],
    num_augmentations: int = 3,
) -> list[tuple[np.ndarray, list[list[float]], list[int]]]:
    """对单张图片应用数据增强，生成多个增强版本。

    Args:
        image: 原始图片 (H, W, C) BGR 格式
        bboxes: YOLO 格式标注框列表
        class_labels: 类别标签列表
        num_augmentations: 每张图片生成的增强版本数

    Returns:
        增强后的 (图片, 标注框, 类别标签) 列表
    """
    transform = create_classroom_augmentation()
    results = []

    for _ in range(num_augmentations):
        try:
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            results.append((
                augmented["image"],
                augmented["bboxes"],
                augmented["class_labels"],
            ))
        except Exception:
            # 某些增强组合可能导致标注框消失，跳过
            continue

    return results


def augment_dataset(
    dataset_dir: Path,
    output_dir: Path,
    num_augmentations: int = 3,
) -> None:
    """对整个数据集应用数据增强。

    Args:
        dataset_dir: 原始数据集目录
        output_dir: 增强后输出目录
        num_augmentations: 每张图片增强数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

    print(f"对 {len(image_files)} 张图片进行数据增强（每张 ×{num_augmentations}）...")

    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # 读取 YOLO 标注
        label_path = labels_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        if not bboxes:
            continue

        # 先复制原始图片和标注
        shutil.copy2(img_path, out_images / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_labels / f"{img_path.stem}.txt")

        # 生成增强版本
        augmented = apply_augmentation(
            image, bboxes, class_labels, num_augmentations
        )

        for i, (aug_img, aug_bboxes, aug_labels) in enumerate(augmented):
            if not aug_bboxes:
                continue

            aug_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(out_images / f"{aug_name}.jpg"), aug_img)

            with open(out_labels / f"{aug_name}.txt", "w") as f:
                for cls, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

    print(f"增强完成，输出目录: {output_dir}")


if __name__ == "__main__":
    import shutil

    dataset_dir = Path("./datasets/cat_yolo")
    output_dir = Path("./datasets/cat_yolo_augmented")

    augment_dataset(dataset_dir, output_dir, num_augmentations=3)
