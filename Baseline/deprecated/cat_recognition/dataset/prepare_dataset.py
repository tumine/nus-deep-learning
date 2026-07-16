"""数据集准备脚本

从 COCO 2017 数据集中提取猫（cat）类别的标注，生成 YOLOv8 格式的训练数据。

COCO 中猫的 category_id = 15（80-class 索引体系）。
"""

import json
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def coco_to_yolo(
    coco_annotation_path: Path,
    output_dir: Path,
    image_dir: Path,
    cat_category_id: int = 15,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """将 COCO 格式的猫类别标注转换为 YOLOv8 格式。

    Args:
        coco_annotation_path: COCO annotations/instances_train2017.json 路径
        output_dir: YOLO 格式数据集输出目录
        image_dir: COCO 图片目录
        cat_category_id: COCO 中猫的类别 ID（80-class 体系为 15）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    # 加载 COCO 标注
    print(f"加载 COCO 标注: {coco_annotation_path}")
    with open(coco_annotation_path, "r") as f:
        coco = json.load(f)

    # 构建 image_id → file_name 映射
    image_id_to_info = {
        img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]}
        for img in coco["images"]
    }

    # 筛选猫类别的标注
    cat_annotations = [
        ann for ann in coco["annotations"]
        if ann["category_id"] == cat_category_id
    ]

    # 按 image_id 分组
    image_to_annotations: dict[int, list[dict]] = {}
    for ann in cat_annotations:
        image_id = ann["image_id"]
        if image_id in image_id_to_info:
            image_to_annotations.setdefault(image_id, []).append(ann)

    print(f"共找到 {len(image_to_annotations)} 张包含猫的图片，{len(cat_annotations)} 个标注框")

    # 创建输出目录结构
    splits = ["train", "val", "test"]
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 随机打乱并划分数据集
    import random
    random.seed(42)
    image_ids = list(image_to_annotations.keys())
    random.shuffle(image_ids)

    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split_map = {}
    for i, img_id in enumerate(image_ids):
        if i < n_train:
            split_map[img_id] = "train"
        elif i < n_train + n_val:
            split_map[img_id] = "val"
        else:
            split_map[img_id] = "test"

    print(f"训练集: {n_train}, 验证集: {n_val}, 测试集: {n_total - n_train - n_val}")

    # 转换标注
    for img_id, annotations in tqdm(image_to_annotations.items(), desc="转换标注"):
        info = image_id_to_info[img_id]
        split = split_map[img_id]
        file_stem = Path(info["file_name"]).stem
        img_w, img_h = info["width"], info["height"]

        # 复制图片
        src_img = image_dir / info["file_name"]
        dst_img = output_dir / "images" / split / info["file_name"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        # 生成 YOLO 格式标注: class_id x_center y_center width height（归一化）
        yolo_lines = []
        for ann in annotations:
            bbox = ann["bbox"]  # COCO: [x, y, width, height]（绝对坐标）
            x, y, w, h = bbox

            # 归一化
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # YOLO 格式：class_id=0（单类别 cat）
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 写入标注文件
        label_path = output_dir / "labels" / split / f"{file_stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    # 生成 dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "cat"},
        "nc": 1,
    }

    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n数据集准备完成: {output_dir}")
    print(f"配置文件: {output_dir / 'dataset.yaml'}")


def download_coco_if_needed(coco_dir: Path) -> tuple[Path, Path]:
    """检查并下载 COCO 2017 数据集。"""
    import urllib.request
    import zipfile

    coco_dir.mkdir(parents=True, exist_ok=True)

    # 图片
    images_dir = coco_dir / "train2017"
    images_zip = coco_dir / "train2017.zip"
    if not images_dir.exists():
        url = "http://images.cocodataset.org/zips/train2017.zip"
        print(f"下载 COCO train2017 图片: {url}")
        print("（如果已有 COCO 数据集，请将图片放到 {images_dir}）")
        print("提示：可使用以下命令手动下载：")
        print(f"  wget {url} -O {images_zip}")
        print(f"  unzip {images_zip} -d {coco_dir}")
        raise FileNotFoundError(f"请手动将 COCO train2017 图片放置到 {images_dir}")

    # 标注文件
    annotations_dir = coco_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    ann_file = annotations_dir / "instances_train2017.json"
    if not ann_file.exists():
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"下载 COCO 标注: {url}")
        print(f"提示：可使用以下命令手动下载：")
        print(f"  wget {url} -O {coco_dir / 'annotations.zip'}")
        print(f"  unzip {coco_dir / 'annotations.zip'} -d {coco_dir}")
        raise FileNotFoundError(f"请手动将 COCO 标注放置到 {ann_file}")

    return images_dir, ann_file


def prepare_custom_dataset(custom_dir: Path, output_dir: Path) -> None:
    """准备自定义猫数据集（教室场景）。

    用户可将教室中拍摄的猫照片放入 custom_dir，标注为 YOLO 格式。

    目录结构：
        custom_cat/
        ├── images/
        │   ├── cat_001.jpg
        │   └── cat_002.jpg
        └── labels/
            ├── cat_001.txt   # YOLO 格式标注
            └── cat_002.txt
    """
    custom_images = custom_dir / "images"
    custom_labels = custom_dir / "labels"

    if not custom_images.exists() or not any(custom_images.iterdir()):
        print(f"\n[提示] 未找到自定义数据集: {custom_dir}")
        print(f"如需添加教室场景的猫图片，请创建目录并放入图片和 YOLO 格式标注：")
        print(f"  图片目录: {custom_images}")
        print(f"  标注目录: {custom_labels}")
        return

    # 将自定义数据合并到训练集
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"

    count = 0
    for img_file in custom_images.iterdir():
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
            label_file = custom_labels / f"{img_file.stem}.txt"
            shutil.copy2(img_file, train_img_dir / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, train_lbl_dir / f"{img_file.stem}.txt")
                count += 1

    print(f"已合并 {count} 张自定义猫图片到训练集")


def main() -> None:
    config = load_config()

    coco_path = Path(config["dataset"]["coco_path"])
    custom_path = Path(config["dataset"]["custom_path"])
    output_dir = Path("./datasets/cat_yolo")

    # 步骤 1: 检查 COCO 数据集
    try:
        images_dir, ann_file = download_coco_if_needed(coco_path)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n请先下载 COCO 2017 数据集后重新运行。")
        return

    # 步骤 2: 转换 COCO 猫类别 → YOLOv8 格式
    print("\n=== 步骤 1: 转换 COCO 猫类别为 YOLO 格式 ===")
    coco_to_yolo(
        coco_annotation_path=ann_file,
        output_dir=output_dir,
        image_dir=images_dir,
        cat_category_id=15,  # COCO 80-class 体系中猫的 ID
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
    )

    # 步骤 3: 合并自定义数据集（如果有）
    print("\n=== 步骤 2: 合并自定义数据集 ===")
    prepare_custom_dataset(custom_path, output_dir)

    print("\n数据集准备完成！")
    print(f"数据集路径: {output_dir}")
    print(f"配置文件: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
