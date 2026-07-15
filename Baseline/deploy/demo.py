"""
端到端部署演示
==============

在本地同时启动 TCP 服务器 + 客户端，演示完整的
"网络传输图片 → 推理 → 返回结果"流水线。

用法：
    python deploy/demo.py --model outputs/.../best_model.pth --image cat.jpg

    # 批量演示
    python deploy/demo.py --model best_model.pth --dir ./test_images/
"""

import argparse
import logging
import socket
import sys
import threading
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("demo")


def _run_server_in_thread(model_path: str, backend: str, port: int) -> tuple[threading.Thread, object]:
    """在后台线程启动 TCP 推理服务器，返回 (thread, server)。"""
    from deploy.tcp_server import TcpBreedServer

    server = TcpBreedServer(
        model_path=model_path,
        backend=backend,
        port=port,
        host="127.0.0.1",
    )

    def _server_main():
        server._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server._server_sock.bind(("127.0.0.1", port))
        server._server_sock.listen(5)
        server._server_sock.settimeout(1.0)
        server._stats["start_time"] = time.time()

        try:
            server._accept_loop()
        except Exception:
            pass

    thread = threading.Thread(target=_server_main, daemon=True)
    thread.start()

    # 等待服务就绪
    time.sleep(1.5)
    logger.info("TCP 服务已就绪 ✅\n")
    return thread, server


def demo_single_image(port: int, image_path: str):
    """单张图片演示。"""
    from deploy.tcp_client import TcpBreedClient, print_result

    path = Path(image_path)
    logger.info(f"{'='*55}")
    logger.info(f"📷 单张推理演示")
    logger.info(f"   图片: {path.name}")
    logger.info(f"{'='*55}")

    t_start = time.time()

    client = TcpBreedClient("127.0.0.1", port)
    result = client.predict_file(str(path))
    client.close()

    total_time = (time.time() - t_start) * 1000
    print_result(result, path.name)
    print(f"  🌐 端到端耗时（含网络）: {total_time:.1f}ms")
    print(f"  ⚡ 纯推理耗时（服务端）: {result['latency_ms']:.1f}ms")
    if "latency_ms" in result:
        print(f"  📡 网络开销:           {total_time - result['latency_ms']:.1f}ms")


def demo_batch(port: int, dir_path: str):
    """批量图片演示。"""
    from deploy.tcp_client import TcpBreedClient

    dir_path = Path(dir_path)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(
        p for p in dir_path.iterdir() if p.suffix.lower() in image_exts
    )

    logger.info(f"{'='*55}")
    logger.info(f"📦 批量推理演示 ({len(image_paths)} 张)")
    logger.info(f"{'='*55}")

    t_start = time.time()

    with TcpBreedClient("127.0.0.1", port) as client:
        for i, img_path in enumerate(image_paths):
            result = client.predict_file(str(img_path))
            logger.info(
                f"  [{i+1}/{len(image_paths)}] {img_path.name:<30} → "
                f"{result['class_name_cn']:<8} {result['confidence']*100:.1f}% "
                f"({result['latency_ms']:.1f}ms)"
            )

    elapsed = time.time() - t_start
    logger.info(f"\n  ✅ 完成: {len(image_paths)} 张, 总耗时 {elapsed:.1f}s")
    logger.info(f"  平均: {elapsed/len(image_paths)*1000:.0f}ms/张（含网络）")


def main():
    parser = argparse.ArgumentParser(
        description="猫品种分类 — 端到端部署演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "torchscript", "onnx"])
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片文件夹路径")
    parser.add_argument("--port", type=int, default=9000, help="TCP 端口")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("请指定 --image 或 --dir")

    logger.info("=" * 55)
    logger.info("🐱 端到端部署演示")
    logger.info("=" * 55)
    logger.info(f"  模型:     {args.model}")
    logger.info(f"  后端:     {args.backend}")
    logger.info(f"  TCP 端口: {args.port}")

    # 1. 后台启动 TCP 服务
    thread, server = _run_server_in_thread(args.model, args.backend, args.port)

    try:
        # 2. 运行演示
        if args.image:
            demo_single_image(args.port, args.image)
        elif args.dir:
            demo_batch(args.port, args.dir)
    finally:
        # 3. 清理
        logger.info("\n停止服务...")
        server._running = False
        if server._server_sock:
            server._server_sock.close()
        thread.join(timeout=3)
        logger.info("演示结束")


if __name__ == "__main__":
    main()
