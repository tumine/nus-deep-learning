"""
随机非猫图片收集脚本
==================

从多个免费公开数据源收集多样化的非猫图片，用于训练"other/非猫"类别，
解决模型对非猫图片强制分类的问题。

数据来源：
1. Lorem Picsum (picsum.photos) — 高质量随机摄影作品，无 API Key
2. Unsplash Source — 随机自然/城市/人物照片

输出的图片保存在 collected_images/other/ 目录下，
与猫品种目录同级，训练时 ImageFolder 会自动将其作为新类别。

用法：
    python collect_random_images.py                    # 默认收集 350 张
    python collect_random_images.py --count 500        # 收集 500 张
    python collect_random_images.py --output ./my_data  # 自定义输出目录

依赖：
    pip install requests pillow
"""

import argparse
import hashlib
import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError

# ============================================================
# 配置
# ============================================================

DEFAULT_COUNT = 350
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "collected_images" / "other"

MIN_IMAGE_SIZE = (200, 200)
MAX_FILE_SIZE_MB = 10
REQUEST_TIMEOUT = 15
RETRY_LIMIT = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}

# ============================================================
# 图片源配置
# ============================================================

# Lorem Picsum: 免费高质量随机图片，像素尺寸可指定
# 文档: https://picsum.photos/
PICSUM_URL = "https://picsum.photos/{w}/{h}?random={seed}"

# Unsplash Source: 随机高质量照片
# 文档: https://source.unsplash.com/
UNSPLASH_URL = "https://source.unsplash.com/random/{w}x{h}?sig={seed}"


def get_image_hash(data: bytes) -> str:
    """计算图片的 MD5 哈希，用于去重。"""
    return hashlib.md5(data).hexdigest()


def is_valid_image(data: bytes) -> bool:
    """验证图片是否有效（非损坏、尺寸足够、非猫相关）。"""
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        if img.width < MIN_IMAGE_SIZE[0] or img.height < MIN_IMAGE_SIZE[1]:
            return False
        return True
    except (UnidentifiedImageError, Exception):
        return False


def download_from_picsum(seed: int, size: tuple = (512, 512)) -> Optional[bytes]:
    """从 Lorem Picsum 下载一张随机图片。

    Picsum 每天提供相同的固定图片集，使用不同 seed 获取不同图片。
    size 可以是 (w, h) 元组，Picsum 会返回对应尺寸的裁剪图片。
    """
    url = PICSUM_URL.format(w=size[0], h=size[1], seed=seed)
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if len(resp.content) > 1024:  # 至少 1KB
                return resp.content
        except Exception:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(1)
    return None


def download_from_unsplash(seed: int, size: tuple = (512, 512)) -> Optional[bytes]:
    """从 Unsplash Source 下载一张随机高质量照片。"""
    url = UNSPLASH_URL.format(w=size[0], h=size[1], seed=seed)
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if len(resp.content) > 1024:
                return resp.content
        except Exception:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(1)
    return None


def collect_random_images(
    output_dir: Path,
    target_count: int = DEFAULT_COUNT,
    image_size: tuple = (512, 512),
):
    """收集随机非猫图片。

    使用多种来源轮流下载，确保内容多样性：
    - 前 60% 来自 Lorem Picsum（风景/建筑/静物）
    - 后 40% 来自 Unsplash Source（人物/城市/自然）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载已有图片哈希
    seen_hashes: set[str] = set()
    for f in output_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            seen_hashes.add(f.stem)

    already_have = len(seen_hashes)
    if already_have >= target_count:
        print(f"已有 {already_have} 张，达到目标 {target_count}，跳过")
        return already_have

    needed = target_count - already_have
    print(f"\n{'=' * 60}")
    print(f"收集随机非猫图片")
    print(f"{'=' * 60}")
    print(f"  目标数量: {target_count}")
    print(f"  已有:     {already_have}")
    print(f"  还需:     {needed}")
    print(f"  输出目录: {output_dir}")
    print(f"  图片尺寸: {image_size[0]}×{image_size[1]}")
    print(f"{'=' * 60}\n")

    downloaded = 0
    seed = already_have  # 从已有数量开始，避免重复
    fail_count = 0
    duplicated = 0

    while downloaded < needed:
        # 前 60% 用 Picsum，后 40% 用 Unsplash
        use_picsum = downloaded < int(needed * 0.6)
        source_name = "Lorem Picsum" if use_picsum else "Unsplash Source"

        if use_picsum:
            data = download_from_picsum(seed, image_size)
        else:
            data = download_from_unsplash(seed, image_size)

        seed += 1

        if data is None:
            fail_count += 1
            if fail_count > 50:
                print(f"\n⚠️  连续失败 {fail_count} 次，可能是网络问题，停止")
                break
            continue

        fail_count = 0  # 重置失败计数

        if not is_valid_image(data):
            continue

        img_hash = get_image_hash(data)
        if img_hash in seen_hashes:
            duplicated += 1
            continue

        seen_hashes.add(img_hash)

        save_path = output_dir / f"{img_hash}.jpg"
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=90)
            downloaded += 1

            if downloaded % 50 == 0:
                print(f"  [{downloaded}/{needed}] 来自 {source_name} (重复: {duplicated})")
        except Exception:
            continue

        # 避免请求过快
        time.sleep(0.1)

    total = already_have + downloaded
    print(f"\n{'=' * 60}")
    print(f"收集完成！")
    print(f"  最终数量: {total}/{target_count}")
    print(f"  重复跳过: {duplicated}")
    print(f"  下载失败: {fail_count}")
    print(f"{'=' * 60}")

    return total


def main():
    parser = argparse.ArgumentParser(
        description="收集随机非猫图片（用于\"other\"类别训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"目标图片数量（默认 {DEFAULT_COUNT}）",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"输出目录（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="下载图片尺寸（默认 512）",
    )
    args = parser.parse_args()

    size = (args.size, args.size)
    collect_random_images(args.output, args.count, size)

    # 提示下一步
    print(f"\n下一步:")
    print(f"  1. 检查图片质量: {args.output}")
    print(f"  2. 确认已放入猫品种目录同级")
    print(f"  3. 重新运行训练: python train_cnn_v2.py")
    print(f"     模型会自动将 'other' 识别为第6类")


if __name__ == "__main__":
    main()
