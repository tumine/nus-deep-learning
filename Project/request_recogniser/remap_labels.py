"""
remap_labels.py

将 request_dataset 下所有标签文件的类别编号从旧映射 (data.yaml)
转换为 best.pt 模型的类别映射。

旧映射 (request_dataset/data.yaml):
    0: block
    1: pencil
    2: eraser

新映射 (best.pt model.names):
    0: block
    1: eraser
    2: pencil

转换规则:  old 0→new 0, old 1→new 2, old 2→new 1

用法（从仓库根目录运行）:
    conda run -n sws python Project/request_recogniser/remap_labels.py
"""

import os
import shutil
from pathlib import Path

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "request_dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
SPLITS = ["train", "valid", "test"]

# 旧 → 新 类别编号映射
# old 0 (block)   → new 0 (block)
# old 1 (pencil)  → new 2 (pencil)
# old 2 (eraser)  → new 1 (eraser)
OLD_TO_NEW = {0: 0, 1: 2, 2: 1}

# 新 data.yaml 类别顺序 (对齐 best.pt)
NEW_NAMES = {0: "block", 1: "eraser", 2: "pencil"}


def backup_data_yaml() -> Path:
    """备份原始 data.yaml。"""
    backup_path = DATA_YAML.with_suffix(".yaml.bak")
    if not backup_path.exists():
        shutil.copy2(DATA_YAML, backup_path)
        print(f"  已备份: {backup_path.name}")
    return backup_path


def update_data_yaml() -> None:
    """将 data.yaml 中的 names 更新为 best.pt 的顺序。"""
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            new_lines.append("names:\n")
            for idx, name in NEW_NAMES.items():
                new_lines.append(f"  {idx}: {name}\n")
            continue
        if in_names:
            if stripped.startswith(("0:", "1:", "2:")) or stripped == "":
                continue
            else:
                in_names = False
        new_lines.append(line)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"  已更新: {DATA_YAML.name}")


def remap_label_file(filepath: Path) -> int:
    """
    转换单个标签文件中的类别编号。
    返回修改的行数。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    changed = 0
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue

        parts = line.split()
        if not parts:
            new_lines.append(line)
            continue

        try:
            old_cls = int(parts[0])
        except ValueError:
            new_lines.append(line)
            continue

        if old_cls in OLD_TO_NEW:
            new_cls = OLD_TO_NEW[old_cls]
            if new_cls != old_cls:
                changed += 1
            parts[0] = str(new_cls)
        # 如果类别不在映射表中，保持原样

        new_lines.append(" ".join(parts))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    return changed


def main() -> None:
    print("=" * 60)
    print("request_dataset 标签类别编号转换")
    print("=" * 60)
    print()
    print("映射规则:")
    print(f"  old 0 (block)  → new 0 (block)")
    print(f"  old 1 (pencil) → new 2 (pencil)")
    print(f"  old 2 (eraser) → new 1 (eraser)")
    print()

    # 1. 备份 data.yaml
    print("[1/2] 备份并更新 data.yaml ...")
    backup_data_yaml()
    update_data_yaml()

    # 2. 转换所有标签文件
    print("\n[2/2] 转换标签文件 ...")
    total_files = 0
    total_changed = 0

    for split in SPLITS:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.is_dir():
            print(f"  ⚠ 跳过（目录不存在）: {labels_dir}")
            continue

        txt_files = sorted(labels_dir.glob("*.txt"))
        split_changed = 0

        for txt_file in txt_files:
            changed = remap_label_file(txt_file)
            if changed > 0:
                split_changed += changed
            total_files += 1

        print(f"  {split}: {len(txt_files)} 个文件, 修改 {split_changed} 行")
        total_changed += split_changed

    print()
    print(f"  总计: {total_files} 个文件, 修改 {total_changed} 行")
    print()
    print("[OK] 转换完成。如需回滚，请恢复 data.yaml.bak 并 git checkout 标签文件。")


if __name__ == "__main__":
    main()
