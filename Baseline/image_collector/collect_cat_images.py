"""
猫图片自动搜集脚本
使用 Selenium 从 Bing Images 搜集 5 种猫的图片，每种 350 张。

猫品种：
- 布偶猫 ragdolls
- 新加坡猫 singapura cats
- 波斯猫 Persian cats
- 斯芬克斯猫 Sphynx cats
- 兔狲 Pallas cats

依赖：
    pip install selenium requests pillow

需要安装 Chrome 浏览器和对应版本的 ChromeDriver：
    https://chromedriver.chromium.org/
    或使用 webdriver-manager 自动管理驱动：
    pip install webdriver-manager

用法：
    python collect_cat_images.py                    # 搜集全部 5 种猫
    python collect_cat_images.py --breed ragdoll    # 只搜集布偶猫
    python collect_cat_images.py --count 100        # 每种搜集 100 张
"""

import argparse
import hashlib
import io
import os
import re
import shutil
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError
from selenium import webdriver
from selenium.common import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ============================================================
# 配置
# ============================================================

# 猫品种及多组搜索关键词（扩大搜索范围）
CAT_BREED_QUERIES: dict[str, list[str]] = {
    "ragdoll": [
        "ragdoll cat",
        "ragdoll cat cute",
        "ragdoll kitten",
        "ragdoll cat face",
        "ragdoll cat portrait",
    ],
    "singapura": [
        "singapura cat",
        "singapura cat cute",
        "singapura kitten",
        "singapura cat face",
        "singapura cat portrait",
    ],
    "persian": [
        "Persian cat",
        "Persian cat cute",
        "Persian kitten",
        "Persian cat face",
        "Persian cat portrait",
    ],
    "sphynx": [
        "Sphynx cat",
        "Sphynx cat cute",
        "Sphynx kitten",
        "hairless cat sphynx",
        "Sphynx cat face",
    ],
    "pallas": [
        "Pallas cat",
        "Pallas cat wild",
        "Pallas cat cute",
        "manul cat",
        "Pallas cat face",
    ],
}

# 每个关键词搜索的翻页数（Bing 每页约 35 张）
PAGES_PER_QUERY = 5

TARGET_COUNT_PER_BREED = 350
OUTPUT_ROOT = Path(__file__).resolve().parent / "collected_images"

# 图片要求
MIN_IMAGE_SIZE = (200, 200)  # 最小宽度、高度
MAX_FILE_SIZE_MB = 10  # 最大文件大小
REQUEST_TIMEOUT = 15  # 下载超时（秒）
SCROLL_PAUSE = 1.5  # 滚动间隔（秒）
MAX_SCROLL_ATTEMPTS = 80  # 单页最大滚动次数（减少等待时间）

# User-Agent（模拟正常浏览器）
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


# ============================================================
# 工具函数
# ============================================================


def get_image_hash(data: bytes) -> str:
    """计算图片数据的 MD5 哈希，用于去重。"""
    return hashlib.md5(data).hexdigest()


def is_valid_image(data: bytes) -> bool:
    """验证图片是否有效且尺寸足够。"""
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


def download_image(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[bytes]:
    """下载图片，返回原始字节数据。"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
        ):
            return None
        data = b""
        for chunk in resp.iter_content(chunk_size=8192):
            data += chunk
            if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
                return None
        return data if data else None
    except Exception:
        return None


# ============================================================
# 核心搜集逻辑
# ============================================================


class CatImageCollector:
    """使用 Selenium 从 Bing Images 搜集猫图片。"""

    def __init__(
        self,
        output_root: Path = OUTPUT_ROOT,
        headless: bool = True,
        driver_path: Optional[str] = None,
    ):
        self.output_root = Path(output_root)
        self.headless = headless
        self.driver_path = driver_path
        self.driver: Optional[webdriver.Chrome] = None
        self.seen_hashes: set[str] = set()

    def _init_driver(self):
        """初始化 Chrome WebDriver。"""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")

        if self.driver_path:
            service = Service(executable_path=self.driver_path)
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            try:
                from webdriver_manager.chrome import ChromeDriverManager

                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except ImportError:
                self.driver = webdriver.Chrome(options=options)

        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

    def _build_bing_url(self, query: str, first: int = 1) -> str:
        """构建 Bing Images 搜索 URL，支持翻页。"""
        params = urllib.parse.urlencode(
            {"q": query, "form": "IRFLTR", "first": str(first)}
        )
        return f"https://www.bing.com/images/search?{params}"

    def _extract_murls_from_page(self) -> list[str]:
        """
        从当前页面提取所有图片的原始链接（murl），不收集缩略图 URL。

        只提取 a.iusc 元素 m 属性中的 murl（原始高清图片链接），
        跳过缩略图 URL（th.bing.com），因为缩略图尺寸太小无法通过验证。
        """
        murls: list[str] = []

        iusc_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.iusc")
        for el in iusc_elements:
            m_attr = el.get_attribute("m") or ""
            murl_match = re.search(r'"murl"\s*:\s*"([^"]+)"', m_attr)
            if murl_match:
                murl = murl_match.group(1)
                if murl.startswith("http") and murl not in murls:
                    murls.append(murl)

        return murls

    def _collect_murls_for_query(self, query: str, pages: int = PAGES_PER_QUERY) -> list[str]:
        """
        对单个搜索关键词进行多页翻页，收集所有 murl。

        使用 Bing 的 first 参数翻页（每页约 35 张），
        每页滚动加载以确保所有缩略图渲染完毕。
        """
        all_murls: list[str] = []

        for page in range(pages):
            first = page * 35 + 1  # Bing 分页参数

            url = self._build_bing_url(query, first=first)
            self.driver.get(url)
            time.sleep(2)

            # 滚动加载当前页的图片
            last_count = 0
            no_new_count = 0

            for _ in range(MAX_SCROLL_ATTEMPTS):
                murls = self._extract_murls_from_page()
                for u in murls:
                    if u not in all_murls:
                        all_murls.append(u)

                if len(all_murls) == last_count:
                    no_new_count += 1
                    if no_new_count >= 5:
                        break
                else:
                    no_new_count = 0
                    last_count = len(all_murls)

                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(SCROLL_PAUSE)

            # 如果本页没拿到新图，说明翻页到底了
            page_murls = [u for u in murls if u in all_murls]
            if len(page_murls) < 3 and page > 0:
                break

        return all_murls

    def _download_batch(
        self,
        murls: list[str],
        save_dir: Path,
        needed: int,
        stats: dict,
    ) -> int:
        """
        批量下载图片。

        Args:
            murls: 图片原始 URL 列表
            save_dir: 保存目录
            needed: 还需要下载的数量
            stats: 统计字典（用于记录失败原因）

        Returns:
            成功下载的数量
        """
        downloaded = 0

        for url in murls:
            if downloaded >= needed:
                break

            data = download_image(url)

            if data is None:
                stats["download_failed"] += 1
                continue

            if not is_valid_image(data):
                stats["invalid_image"] += 1
                continue

            img_hash = get_image_hash(data)
            if img_hash in self.seen_hashes:
                stats["duplicate"] += 1
                continue

            self.seen_hashes.add(img_hash)

            save_path = save_dir / f"{img_hash}.jpg"
            try:
                img = Image.open(io.BytesIO(data))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(save_path, "JPEG", quality=90)
                downloaded += 1
                stats["downloaded"] += 1
            except Exception:
                stats["save_failed"] += 1
                continue

            if downloaded % 50 == 0 and downloaded > 0:
                print(f"    已下载 {downloaded}/{needed}")

        return downloaded

    def collect_breed(
        self,
        breed_key: str,
        target_count: int = TARGET_COUNT_PER_BREED,
    ) -> int:
        """
        搜集单个猫品种的图片，使用多关键词 + 多页搜索扩大图片池。

        Args:
            breed_key: 品种键名（如 'ragdoll'）
            target_count: 目标搜集数量

        Returns:
            实际成功下载的图片数量
        """
        queries = CAT_BREED_QUERIES.get(breed_key)
        if not queries:
            print(f"[错误] 未知品种: {breed_key}")
            return 0

        save_dir = self.output_root / breed_key
        save_dir.mkdir(parents=True, exist_ok=True)

        # 加载已有图片的哈希
        existing_hashes: set[str] = set()
        for f in save_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                existing_hashes.add(f.stem)
        self.seen_hashes = existing_hashes.copy()

        already_have = len(existing_hashes)
        if already_have >= target_count:
            print(f"[{breed_key}] 已有 {already_have} 张，达到目标 {target_count}，跳过")
            return already_have

        needed = target_count - already_have
        print(f"\n{'=' * 60}")
        print(f"[{breed_key}] 目标: {target_count} 张")
        print(f"[{breed_key}] 已有: {already_have}, 还需: {needed}")
        print(f"[{breed_key}] 搜索关键词 ({len(queries)} 组):")
        for q in queries:
            print(f"    - \"{q}\"")
        print(f"{'=' * 60}")

        # 统计
        stats: dict[str, int] = defaultdict(int)
        total_murls = 0

        for qi, query in enumerate(queries):
            remaining = target_count - already_have - stats["downloaded"]
            if remaining <= 0:
                break

            print(f"\n  [{breed_key}] 搜索 {qi + 1}/{len(queries)}: \"{query}\"")
            murls = self._collect_murls_for_query(query)
            total_murls += len(murls)
            print(
                f"  [{breed_key}] 收集到 {len(murls)} 个原始图片 URL"
                f"（累计 {total_murls} 个）"
            )

            if not murls:
                continue

            before = stats["downloaded"]
            downloaded = self._download_batch(murls, save_dir, remaining, stats)
            print(
                f"  [{breed_key}] 本轮下载 {downloaded} 张，"
                f"总进度: {already_have + stats['downloaded']}/{target_count}"
            )

        total = already_have + stats["downloaded"]
        print(f"\n[{breed_key}] 完成！")
        print(f"  总计收集 URL: {total_murls}")
        print(f"  成功下载: {stats['downloaded']}")
        print(f"  下载失败: {stats['download_failed']}")
        print(f"  图片无效（尺寸太小等）: {stats['invalid_image']}")
        print(f"  重复图片: {stats['duplicate']}")
        print(f"  保存失败: {stats['save_failed']}")
        print(f"  最终数量: {total}/{target_count}")

        return total

    def run(
        self,
        breeds: Optional[list[str]] = None,
        target_count: int = TARGET_COUNT_PER_BREED,
    ):
        """
        运行搜集流程。

        Args:
            breeds: 要搜集的品种列表，None 表示全部
            target_count: 每种的目标数量
        """
        if breeds is None:
            breeds = list(CAT_BREED_QUERIES.keys())

        print(f"猫图片搜集脚本启动")
        print(f"品种: {breeds}")
        print(f"每种目标: {target_count} 张")
        print(f"输出目录: {self.output_root.resolve()}")
        print()

        self._init_driver()

        try:
            for breed_key in breeds:
                self.collect_breed(breed_key, target_count=target_count)
        finally:
            if self.driver:
                self.driver.quit()
                print("\n浏览器已关闭")

        # 打印总结
        print(f"\n{'=' * 60}")
        print("搜集完成！总结：")
        for breed_key in breeds:
            save_dir = self.output_root / breed_key
            count = (
                len(
                    [
                        f
                        for f in save_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                    ]
                )
                if save_dir.exists()
                else 0
            )
            queries = CAT_BREED_QUERIES[breed_key]
            print(f"  {breed_key} ({queries[0]}): {count} 张")
        print(f"{'=' * 60}")


# ============================================================
# 入口
# ============================================================


def main():
    global PAGES_PER_QUERY

    parser = argparse.ArgumentParser(
        description="从 Bing Images 搜集猫图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python collect_cat_images.py                         # 搜集全部 5 种猫，每种 350 张
  python collect_cat_images.py --breed ragdoll         # 只搜集布偶猫
  python collect_cat_images.py --breed ragdoll persian # 搜集布偶猫和波斯猫
  python collect_cat_images.py --count 100             # 每种搜集 100 张
  python collect_cat_images.py --no-headless           # 显示浏览器窗口
  python collect_cat_images.py --output ./my_images    # 自定义输出目录
  python collect_cat_images.py --pages 8               # 每个关键词搜索 8 页
        """,
    )
    parser.add_argument(
        "--breed",
        nargs="+",
        choices=list(CAT_BREED_QUERIES.keys()),
        default=None,
        help="要搜集的猫品种（默认全部）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=TARGET_COUNT_PER_BREED,
        help=f"每种猫的目标图片数量（默认 {TARGET_COUNT_PER_BREED}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"图片输出目录（默认 {OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="显示浏览器窗口（调试用）",
    )
    parser.add_argument(
        "--driver",
        type=str,
        default=None,
        help="ChromeDriver 可执行文件路径",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=PAGES_PER_QUERY,
        help=f"每个搜索关键词翻页数（默认 {PAGES_PER_QUERY}）",
    )

    args = parser.parse_args()

    # 更新翻页参数
    PAGES_PER_QUERY = args.pages

    collector = CatImageCollector(
        output_root=args.output,
        headless=not args.no_headless,
        driver_path=args.driver,
    )

    collector.run(
        breeds=args.breed,
        target_count=args.count,
    )


if __name__ == "__main__":
    main()
