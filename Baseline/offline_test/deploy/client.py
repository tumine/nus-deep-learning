"""
猫品种识别 API — Python 客户端示例
===================================

演示如何通过 HTTP 调用已部署的推理服务。

用法：
    # 单张预测
    python deploy/client.py --image cat.jpg

    # 文件夹批量预测
    python deploy/client.py --dir ./test_images/ --output results.json

    # 指定服务地址
    python deploy/client.py --image cat.jpg --url http://192.168.1.100:8000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests


class CatBreedClient:
    """猫品种识别 API 客户端。"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CatBreedClient/1.0"})

    def health(self) -> dict:
        """健康检查。"""
        resp = self.session.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def info(self) -> dict:
        """获取模型信息。"""
        resp = self.session.get(f"{self.base_url}/info", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def predict_file(self, image_path: str) -> dict:
        """对单张图片进行预测。"""
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            resp = self.session.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()

    def predict_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> dict:
        """从字节数据进行预测。"""
        files = {"file": (filename, image_bytes, "image/jpeg")}
        resp = self.session.post(
            f"{self.base_url}/predict",
            files=files,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, image_paths: list[str]) -> dict:
        """批量预测。"""
        files = []
        for path in image_paths:
            f = open(path, "rb")
            files.append(("files", (Path(path).name, f, "image/jpeg")))

        try:
            resp = self.session.post(
                f"{self.base_url}/predict/batch",
                files=files,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for _, (_, f, _) in files:
                f.close()


def print_result(result: dict):
    """格式化打印单张预测结果。"""
    print(f"\n{'='*50}")
    print(f"🖼️  预测结果")
    print(f"{'='*50}")
    print(f"  最佳预测: {result['class_name_cn']} ({result['class_name']})")
    print(f"  置信度:   {result['confidence']*100:.2f}%")
    print(f"  推理耗时: {result['latency_ms']:.2f} ms")
    print(f"\n  Top-5 概率分布:")
    for item in result["top5"]:
        bar = "█" * int(item["probability"] * 30)
        marker = " ←" if item["rank"] == 1 else ""
        print(f"  {item['rank']}. {item['class_name_cn']:<8} "
              f"{bar} {item['probability']*100:.1f}%{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="猫品种识别 API 客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="API 服务地址 (默认 http://localhost:8000)")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片文件夹（批量预测）")
    parser.add_argument("--output", type=str, help="结果输出 JSON 路径")
    parser.add_argument("--health", action="store_true", help="健康检查")
    parser.add_argument("--info", action="store_true", help="查看模型信息")
    args = parser.parse_args()

    client = CatBreedClient(args.url)

    # ---- 健康检查 ----
    if args.health:
        result = client.health()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # ---- 模型信息 ----
    if args.info:
        result = client.info()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # ---- 单张预测 ----
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"文件不存在: {image_path}")
            return

        result = client.predict_file(str(image_path))
        print_result(result)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {args.output}")
        return

    # ---- 文件夹批量预测 ----
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"文件夹不存在: {dir_path}")
            return

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in image_exts
        )

        if not image_paths:
            print(f"文件夹中没有图片: {dir_path}")
            return

        print(f"找到 {len(image_paths)} 张图片，开始批量预测...")
        t0 = time.time()

        results = []
        # 每批最多 20 张（API 限制）
        batch_size = 20
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_result = client.predict_batch([str(p) for p in batch])
            results.extend(batch_result["results"])
            print(f"  进度: {min(i+batch_size, len(image_paths))}/{len(image_paths)}")

        elapsed = time.time() - t0
        print(f"\n批量预测完成: {len(results)} 张, 总耗时 {elapsed:.1f}s")

        # 按置信度排序输出摘要
        results_sorted = sorted(results, key=lambda r: r["confidence"], reverse=True)
        for r in results_sorted:
            print(f"  {r['class_name_cn']:<8} {r['confidence']*100:.1f}%  "
                  f"({r['latency_ms']:.1f}ms)")

        if args.output:
            output_data = {
                "total": len(results),
                "elapsed_seconds": round(elapsed, 2),
                "results": results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {args.output}")
        return

    # 无参数时显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
