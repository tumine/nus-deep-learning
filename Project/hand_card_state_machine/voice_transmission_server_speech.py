#!/usr/bin/env python3
"""
麦克风流式传输与哭声识别服务器（PANNs 版本）
==============================================

功能：
  1. 电脑端启动 HTTPS 服务器，手机浏览器打开网页后通过 WebRTC 传输麦克风音频流
  2. 电脑端接收音频流，重采样后传入 PANNs（Pretrained Audio Neural Networks）
     预训练模型识别哭声
  3. 实时输出识别结果到控制台，并通过 SSE 推送到手机端网页

依赖库：
  必需：aiortc, aiohttp, numpy, torch, panns-inference
  可选：scipy（更高质量的音频重采样）、cryptography（SSL 证书生成）

使用方式：
  python voice_transmission_server_panns.py [--port 8080] [--threshold 0.3]
  手机通过 Tailscale 访问：https://<电脑Tailscale-IP>:8080
  （首次访问需在浏览器中接受自签名证书警告）
"""

import argparse
import asyncio
import json
import logging
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

# ============================================================
# 可选依赖：scipy（高质量重采样）
# ============================================================
try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice_server_panns")

# ============================================================
# PANNs 与音频处理常量
# ============================================================
PANNS_SAMPLE_RATE = 32000       # PANNs 模型要求 32kHz 采样率
PANNS_WINDOW = 32000            # 每次推理的样本数 (~1 秒)
PANNS_HOP = 16000               # 滑动窗口步长 (50% 重叠)

# 哭声检测阈值（PANNs 输出的 sigmoid 置信度，0~1）
DEFAULT_CRY_THRESHOLD = 0.3

# 哭声检测冷却时间（秒）：检测到一次哭声后，冷却期间跳过推理
CRY_COOLDOWN_SECONDS = 5.0

# 教师端 WebSocket 地址
DEFAULT_TEACHER_URL = "ws://127.0.0.1:8000/ws"

# 用于在类别表中搜索哭声相关类别的关键词（小写匹配）
CRY_KEYWORDS = ["cry", "sob", "wail", "whimper", "bawl", "howl"]

# 内置的哭声类别索引（当无法从 panns-inference 获取 labels 时的回退值）
# 基于 AudioSet 527 类完整列表
_FALLBACK_CRY_CLASSES = {
    20: "Baby cry, infant cry",
    499: "Crying, sobbing",
}


# ============================================================
# 音频重采样
# ============================================================

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """将音频从原始采样率重采样到目标采样率。

    优先使用 scipy 的多相滤波器重采样（质量好），回退到 numpy 线性插值。
    """
    if orig_sr == target_sr:
        return audio
    if HAS_SCIPY:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return resample_poly(audio, up, down).astype(np.float32)
    else:
        # numpy 线性插值回退方案
        target_len = int(len(audio) * target_sr / orig_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, len(audio) - 1)
        frac = (indices - lo).astype(np.float32)
        return (audio[lo] * (1 - frac) + audio[hi] * frac).astype(np.float32)


# ============================================================
# PANNs 哭声检测器
# ============================================================

class PannsCryDetector:
    """封装 PANNs 模型加载与哭声检测逻辑。

    PANNs（Pretrained Audio Neural Networks）基于 AudioSet 训练，
    可识别 527 类音频事件，包括 "Baby cry, infant cry"、"Crying, sobbing" 等哭声类别。

    本实现使用 panns-inference 库加载 CNN14 预训练模型，对输入音频进行
    clip-level 分类（整个音频片段给出一个概率分布）。
    """

    def __init__(self, threshold: float = DEFAULT_CRY_THRESHOLD):
        self.threshold = threshold
        self.model = None
        self.device = "cpu"
        self.cry_classes: dict[int, str] = {}  # {class_id: display_name}
        self._lock = threading.Lock()
        self._load_model()
        self._load_cry_classes()

    def _download_model_if_needed(self) -> str | None:
        """手动下载 PANNs 模型到本地（绕过 panns-inference 内置的 wget 调用，兼容 Windows）。

        Returns:
            模型文件路径，下载失败返回 None。
        """
        import os
        import urllib.request

        checkpoint_dir = Path.home() / "panns_data"
        checkpoint_path = checkpoint_dir / "Cnn14_mAP=0.431.pth"

        if checkpoint_path.exists():
            logger.info(f"PANNs 模型文件已存在: {checkpoint_path}")
            return str(checkpoint_path)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # PANNs CNN14 预训练模型下载地址
        model_url = (
            "https://zenodo.org/record/3987831/files/"
            "Cnn14_mAP%3D0.431.pth?download=1"
        )

        logger.info(f"PANNs 模型文件不存在，正在用 Python 下载（~500MB，请耐心等待）...")
        logger.info(f"下载地址: {model_url}")
        logger.info(f"保存路径: {checkpoint_path}")

        try:
            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 // total_size)
                    logger.info(f"  下载进度: {pct}% ({downloaded // (1024*1024)}/{total_size // (1024*1024)} MB)")

            urllib.request.urlretrieve(model_url, str(checkpoint_path), reporthook=_progress)
            logger.info("PANNs 模型下载完成。")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            # 清理不完整的下载文件
            if checkpoint_path.exists():
                try:
                    os.remove(str(checkpoint_path))
                except Exception:
                    pass
            return None

    def _load_model(self):
        """从 panns-inference 加载 PANNs 模型（首次运行自动下载）。"""
        logger.info("正在加载 PANNs 模型（首次运行需联网下载，请稍候）...")
        try:
            import torch
            from panns_inference import AudioTagging

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PANNs 使用设备: {self.device}")

            # 先尝试手动下载模型（用 Python urllib，兼容 Windows）
            checkpoint_path = self._download_model_if_needed()

            if checkpoint_path:
                self.model = AudioTagging(checkpoint_path=checkpoint_path, device=self.device)
            else:
                # 回退：让 panns-inference 自行处理（可能失败于 wget）
                logger.warning("手动下载失败，回退到 panns-inference 内置下载...")
                self.model = AudioTagging(checkpoint_path=None, device=self.device)

            logger.info("PANNs 模型加载成功。")
        except ImportError:
            logger.error("缺少 torch 或 panns-inference，请安装：")
            logger.error("  pip install torch panns-inference")
            sys.exit(1)
        except Exception as e:
            logger.error(f"PANNs 模型加载失败: {e}")
            sys.exit(1)

    def _load_cry_classes(self):
        """从 panns-inference 的 labels 列表或 CSV 文件中搜索哭声相关类别。

        加载优先级：
          1. panns_inference.labels（内置列表）
          2. 本地 panns_data/class_labels_indices.csv
          3. 从 Google AudioSet 官方地址自动下载 CSV
          4. 内置回退索引 _FALLBACK_CRY_CLASSES
        """
        import urllib.request

        # 方法 1: panns-inference 内置 labels 列表
        try:
            from panns_inference import labels as panns_labels
            self._parse_labels(panns_labels, "panns_inference.labels")
            if self.cry_classes:
                return
        except Exception:
            pass

        # 方法 2 + 3: 从 CSV 文件加载（本地或下载）
        for csv_src in self._get_csv_sources():
            try:
                if csv_src.startswith("http"):
                    logger.info(f"正在从 {csv_src} 下载类别映射表...")
                    with urllib.request.urlopen(csv_src, timeout=15) as resp:
                        csv_text = resp.read().decode("utf-8")
                    source = csv_src
                else:
                    csv_path = Path(csv_src)
                    if not csv_path.exists():
                        continue
                    logger.info(f"正在从本地 {csv_path} 读取类别映射表...")
                    csv_text = csv_path.read_text(encoding="utf-8")
                    source = str(csv_path)

                labels_list = self._parse_csv_to_labels(csv_text)
                if labels_list:
                    self._parse_labels(labels_list, source)
                    if self.cry_classes:
                        return
            except Exception as e:
                logger.debug(f"尝试 {csv_src} 失败: {e}")
                continue

        # 方法 4: 内置回退索引
        logger.warning("所有方法均无法加载类别映射表，使用内置回退索引。")
        self.cry_classes = dict(_FALLBACK_CRY_CLASSES)
        for cid, name in self.cry_classes.items():
            logger.info(f"  [{cid}] {name}")

    @staticmethod
    def _get_csv_sources() -> list[str]:
        """返回 CSV 文件来源列表（本地路径 + 远程 URL）。"""
        return [
            # 本地路径：panns_data 目录
            "panns_data/class_labels_indices.csv",
            str(Path.home() / ".cache/panns/class_labels_indices.csv"),
            # 官方下载地址
            "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
            # GitHub 镜像
            "https://raw.githubusercontent.com/bpiyush/PANNs/master/metadata/class_labels_indices.csv",
            "https://raw.githubusercontent.com/IBM/audioset-classification/master/audioset_classify/metadata/class_labels_indices.csv",
        ]

    @staticmethod
    def _parse_csv_to_labels(csv_text: str) -> list[str] | None:
        """将 CSV 文本解析为 527 个类别名列表。"""
        labels = []
        for line in csv_text.strip().split("\n")[1:]:  # 跳过表头
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            # 去掉引号
            name = parts[2].strip().strip('"')
            labels.append(name)
        return labels if len(labels) >= 500 else None  # 至少要有 500+ 类才有效

    def _parse_labels(self, labels_list: list[str], source: str):
        """遍历类别名列表，搜索哭声关键词。"""
        for idx, name in enumerate(labels_list):
            name_lower = name.lower()
            if any(kw in name_lower for kw in CRY_KEYWORDS):
                if idx not in self.cry_classes:
                    self.cry_classes[idx] = name
        if self.cry_classes:
            logger.info(f"从 [{source}] 发现 {len(self.cry_classes)} 个哭声相关类别:")
            for cid, name in self.cry_classes.items():
                logger.info(f"  [{cid}] {name}")

    def predict(self, waveform: np.ndarray) -> dict:
        """对一段 32kHz 单声道音频进行 PANNs 推理。

        Args:
            waveform: float32 numpy array, shape (N,), 32kHz, 值域 [-1, 1]

        Returns:
            dict:
              - cry_detected: bool        是否检测到哭声
              - max_cry_score: float      哭声类别最高得分
              - cry_details: list[dict]   超过阈值的类别详情
              - top_classes: list[str]    整体 Top-5 类别名
        """
        import torch

        with self._lock:
            # 输入过短时补零
            if len(waveform) < PANNS_WINDOW:
                padded = np.zeros(PANNS_WINDOW, dtype=np.float32)
                padded[: len(waveform)] = waveform
                waveform = padded

            # PANNs 输入需要 (batch_size, samples)
            audio_batch = waveform[None, :]  # (1, N)

            # 推理（clip-level 概率）
            with torch.no_grad():
                clipwise_output, _ = self.model.inference(audio_batch)
                # panns-inference 某些版本直接返回 numpy 数组，兼容处理
                out = clipwise_output[0]
                if hasattr(out, 'cpu'):
                    scores = out.cpu().numpy()  # torch tensor
                else:
                    scores = np.asarray(out)     # 已是 numpy 数组

        result = {
            "cry_detected": False,
            "max_cry_score": 0.0,
            "cry_details": [],
            "top_classes": [],
        }

        # 检查哭声类别
        for class_id, class_name in self.cry_classes.items():
            if class_id >= scores.shape[0]:
                continue
            class_score = float(scores[class_id])
            if class_score > result["max_cry_score"]:
                result["max_cry_score"] = class_score
            if class_score >= self.threshold:
                result["cry_detected"] = True
                result["cry_details"].append({
                    "class": class_name,
                    "class_id": int(class_id),
                    "score": class_score,
                    "frame": 0,  # PANNs 为 clip-level，无帧概念，用 0 占位
                })

        # 整体 Top-5 类别
        top5_idx = np.argsort(scores)[-5:][::-1]
        result["top_classes"] = [
            {"id": int(i),
             "name": self.cry_classes.get(int(i), f"class_{int(i)}"),
             "score": round(float(scores[i]), 4)}
            for i in top5_idx
        ]

        return result


# ============================================================
# 音频流处理器
# ============================================================

class AudioProcessor:
    """从 WebRTC AudioFrame 接收音频，缓冲后送入 PANNs 推理。

    音频处理流水线：
      WebRTC Opus(48kHz) → PCM int16 → float32[-1,1] → 重采样到 32kHz → 缓冲 → PANNs 推理
    """

    def __init__(self, detector: PannsCryDetector | None, on_result=None,
                 speech_detector=None):
        self.detector = detector
        self.on_result = on_result  # 回调函数，接收推理结果 dict
        self.speech_detector = speech_detector
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = None
        self.channels = None

        # 统计
        self.total_samples = 0
        self.inference_count = 0
        self.cry_events = 0
        self.start_time = time.time()
        self.last_cry_time = 0.0  # 最后一次检测到哭声的时间戳
        self.last_max_cry_score = 0.0  # 上一次推理的哭声最高分（冷却期间保留显示）

    def add_frame(self, audio_frame) -> bool:
        """添加一个 WebRTC 音频帧到缓冲区。

        aiortc 底层通过 av 库解码 Opus，AudioFrame 的 API:
          - .sample_rate      采样率 (int)
          - .layout.nb_channels  声道数 (int; layout.name 如 'mono'/'stereo')
          - .to_ndarray()     返回 (channels, samples), dtype=int16

        Returns:
            True 如果缓冲区已积累足够样本可以推理
        """
        # to_ndarray() 返回 (channels, samples), dtype=int16
        audio_array = audio_frame.to_ndarray()

        # 从 audio_frame.layout 获取声道数
        num_channels = audio_frame.layout.nb_channels

        if self.sample_rate is None:
            self.sample_rate = audio_frame.sample_rate
            self.channels = num_channels
            logger.info(
                f"音频流已建立: {self.sample_rate} Hz, {self.channels} 声道"
            )

        # 转为单声道
        if num_channels >= 2:
            mono = audio_array.mean(axis=0).astype(np.int16)
        else:
            mono = audio_array[0].astype(np.int16)

        # int16 → float32 归一化到 [-1, 1]
        float_audio = mono.astype(np.float32) / 32768.0

        # 将手机音频送入语音请求检测器。
        # detector 只有在 WAIT_CARD 状态被 enable() 后才会真正缓存与识别。
        if self.speech_detector is not None:
            self.speech_detector.feed_audio(
                float_audio,
                sample_rate=audio_frame.sample_rate,
            )

        # 当前统一启动脚本不需要哭声检测时，到这里直接返回。
        if self.detector is None:
            return False

        # 重采样到 32kHz
        if audio_frame.sample_rate != PANNS_SAMPLE_RATE:
            float_audio = resample_audio(
                float_audio, audio_frame.sample_rate, PANNS_SAMPLE_RATE
            )

        # 追加到缓冲区
        self.buffer = np.concatenate([self.buffer, float_audio])
        self.total_samples += len(float_audio)

        return len(self.buffer) >= PANNS_WINDOW

    def process(self):
        """从缓冲区取一个窗口的数据进行 PANNs 推理并输出结果。"""
        if len(self.buffer) < PANNS_WINDOW:
            return

        # 取窗口数据，滑动步长为 50% 重叠
        waveform = self.buffer[:PANNS_WINDOW]
        self.buffer = self.buffer[PANNS_HOP:]

        # 冷却时间检查：检测到哭声后跳过推理 CRY_COOLDOWN_SECONDS 秒
        now = time.time()
        if (now - self.last_cry_time) < CRY_COOLDOWN_SECONDS:
            # 冷却期间：跳过推理，仅推送心跳状态（保持手机端计数器同步）
            if self.on_result:
                try:
                    self.on_result({
                        "cry_detected": False,
                        "max_cry_score": self.last_max_cry_score,
                        "cry_events": self.cry_events,
                        "inference_count": self.inference_count,
                        "elapsed": round(now - self.start_time, 1),
                        "top_classes": [],
                        "cry_details": [],
                        "cooldown": True,
                    })
                except Exception as e:
                    logger.error(f"SSE 回调出错（冷却期间）: {e}", exc_info=True)
            return

        self.inference_count += 1

        # 推理
        try:
            result = self.detector.predict(waveform)
        except Exception as e:
            logger.error(f"PANNs 推理出错: {e}")
            return

        # 输出结果
        elapsed = time.time() - self.start_time
        result["elapsed"] = round(elapsed, 1)
        result["inference_count"] = self.inference_count

        result["cry_events"] = self.cry_events  # 始终填充，避免 SSE 传 None
        self.last_max_cry_score = result["max_cry_score"]  # 保存上一次的值（供冷却期间推送）
        if result["cry_detected"]:
            self.cry_events += 1
            result["cry_events"] = self.cry_events
            self._print_cry_alert(result)
            self.last_cry_time = time.time()  # 记录最后一次哭声检测时间
        else:
            # 每 10 次推理输出一次心跳
            if self.inference_count % 10 == 0:
                top = ", ".join(
                    f"{c['name']}({c['score']:.2f})" for c in result["top_classes"][:3]
                )
                logger.info(
                    f"#{self.inference_count} | 已运行 {elapsed:.0f}s | "
                    f"Top: {top} | 哭声评分: {result['max_cry_score']:.4f}"
                )

        # 触发回调（推送 SSE 等）
        if self.on_result:
            try:
                self.on_result(result)
            except Exception as e:
                logger.error(f"SSE 回调出错（第 {self.inference_count} 次推理）: {e}", exc_info=True)

    def _print_cry_alert(self, result: dict):
        """检测到哭声时输出醒目的告警信息。"""
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'=' * 60}")
        print(f"  *** 检测到哭声 ***  时间: {ts}  (已运行 {result['elapsed']:.1f}s)")
        print(f"  累计哭声事件: {result['cry_events']}  推理次数: {result['inference_count']}")
        for detail in result["cry_details"]:
            print(
                f"    类别: {detail['class']} (ID={detail['class_id']})  "
                f"得分: {detail['score']:.4f}"
            )
        print(f"  本窗口哭声最高分: {result['max_cry_score']:.4f}")
        print(f"{'=' * 60}\n")


# ============================================================
# SSE 事件推送管理器
# ============================================================

class SSEManager:
    """管理 SSE (Server-Sent Events) 客户端连接，向手机端推送实时检测结果。"""

    def __init__(self):
        self._clients: list[web.StreamResponse] = []
        self._lock = asyncio.Lock()

    async def add_client(self, response: web.StreamResponse):
        async with self._lock:
            self._clients.append(response)

    async def remove_client(self, response: web.StreamResponse):
        async with self._lock:
            if response in self._clients:
                self._clients.remove(response)

    async def broadcast(self, data: dict):
        """向所有 SSE 客户端推送一条消息。"""
        msg = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        async with self._lock:
            dead = []
            for client in self._clients:
                try:
                    await client.write(msg.encode("utf-8"))
                    await client.drain()
                except (ConnectionResetError, RuntimeError):
                    dead.append(client)
            for c in dead:
                self._clients.remove(c)


# ============================================================
# 教师端通知器（WebSocket）
# ============================================================

class TeacherNotifier:
    """通过 WebSocket 向教师端发送哭声检测事件消息。

    教师端 (teacher_client.py) 在 /ws 端点监听 WebSocket 连接，
    接收 JSON 格式的消息并展示在前端页面。
    """

    def __init__(self, teacher_url: str):
        self.teacher_url = teacher_url
        self.ws = None
        self._message_counter = 0

    async def _connect_if_needed(self):
        """如果未连接则尝试建立 WebSocket 连接。"""
        if self.ws is not None:
            return
        try:
            import websockets
            self.ws = await websockets.connect(
                self.teacher_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            logger.info(f"已连接到教师端: {self.teacher_url}")
        except ImportError:
            logger.warning(
                "缺少 websockets 库，无法连接教师端。"
                "安装: pip install websockets"
            )
        except Exception as e:
            logger.warning(f"无法连接到教师端 ({self.teacher_url}): {e}")

    async def send_cry_alert(self):
        """发送哭声检测告警到教师端。

        消息格式：
          - message_id: 唯一消息编号
          - axis_x / axis_y: 无效值（哭声检测无位置信息）
          - request: "教师协助"
          - description: 哭声检测详情
        """
        await self._connect_if_needed()
        if self.ws is None:
            return

        self._message_counter += 1
        msg = {
            "message_id": (
                f"cry-{int(time.time() * 1000)}-"
                f"{self._message_counter:04d}"
            ),
            "axis_x": -1,
            "axis_y": -1,
            "request": "教师协助",
            "description": "检测到婴儿哭声，请前往查看",
        }
        try:
            import websockets
            await self.ws.send(json.dumps(msg, ensure_ascii=False))
            logger.info(f"已向教师端发送哭声告警: {msg['message_id']}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                "教师端 WebSocket 连接已关闭，将在下次检测时重连"
            )
            self.ws = None
        except Exception as e:
            logger.warning(f"向教师端发送消息失败: {e}")
            self.ws = None


# ============================================================
# 手机端网页 (HTML + JS)
# ============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>麦克风音频传输</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f1a; color: #e0e0e0; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center; padding: 20px;
}
h1 { font-size: 1.4rem; margin: 20px 0 6px; color: #fff; text-align: center; }
.subtitle { font-size: 0.82rem; color: #777; margin-bottom: 24px; text-align: center; }
.card {
    background: #1a1a2e; border-radius: 14px; padding: 20px;
    width: 100%; max-width: 380px; margin-bottom: 14px; border: 1px solid #2a2a4a;
}
.status-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.85rem; }
.status-row .label { color: #888; }
.status-row .value { color: #ccc; font-variant-numeric: tabular-nums; }
.dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}
.dot.idle { background: #555; }
.dot.connecting { background: #f0a500; animation: pulse 1.5s infinite; }
.dot.connected { background: #00c853; }
.dot.error { background: #ff1744; }
.dot.cry { background: #ff1744; animation: pulse 0.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.btn {
    display: block; width: 100%; max-width: 380px; padding: 15px; margin: 6px 0;
    border: none; border-radius: 12px; font-size: 1.05rem; font-weight: 600;
    cursor: pointer; transition: transform 0.1s;
}
.btn:active { transform: scale(0.97); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-start { background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; }
.btn-stop { background: #2a2a4a; color: #ff6b6b; border: 1px solid #ff6b6b; }
.alert-box {
    width: 100%; max-width: 380px; padding: 14px; border-radius: 12px;
    background: #2a0a0a; border: 1px solid #ff1744; color: #ff6b6b;
    font-weight: 600; text-align: center; margin-bottom: 14px; display: none;
}
.log-area {
    width: 100%; max-width: 380px; background: #0a0a14; border-radius: 12px;
    padding: 14px; margin-top: 8px; max-height: 240px; overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.72rem;
    color: #888; border: 1px solid #1a1a2e;
}
.log-entry { padding: 2px 0; }
.log-entry.cry { color: #ff6b6b; font-weight: bold; }
.log-entry.info { color: #4fc3f7; }
.cry-stats-area {
    width: 100%; max-width: 380px; background: #0a0a14; border-radius: 12px;
    padding: 14px; margin-bottom: 14px; max-height: 200px; overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.72rem;
    color: #888; border: 1px solid #2a2a4a;
}
.cry-stats-area .cry-stats-title {
    color: #ff6b6b; font-weight: bold; font-size: 0.78rem;
    margin-bottom: 8px; border-bottom: 1px solid #2a1a1a; padding-bottom: 6px;
}
.cry-stats-area .cry-entry {
    padding: 2px 0; color: #ff6b6b; font-weight: bold;
}
.cry-stats-area .cry-entry .time { color: #aa5555; font-weight: normal; }
.cry-stats-empty { color: #444; font-style: italic; }
</style>
</head>
<body>
<h1>🎤 麦克风音频传输</h1>
<p class="subtitle">将手机麦克风音频流发送到电脑进行哭声识别（PANNs）</p>

<div class="alert-box" id="alertBox"></div>

<div class="card">
    <div class="status-row">
        <span class="label">状态</span>
        <span class="value"><span class="dot idle" id="dot"></span><span id="statusText">就绪</span></span>
    </div>
    <div class="status-row"><span class="label">WebRTC 连接</span><span class="value" id="connState">--</span></div>
    <div class="status-row"><span class="label">音频采样率</span><span class="value" id="audioRate">--</span></div>
    <div class="status-row"><span class="label">已传输时长</span><span class="value" id="duration">0s</span></div>
    <div class="status-row"><span class="label">推理次数</span><span class="value" id="infCount">0</span></div>
    <div class="status-row"><span class="label">哭声事件</span><span class="value" id="cryCount">0</span></div>
    <div class="status-row"><span class="label">哭声评分</span><span class="value" id="cryScore">0.0000</span></div>
</div>

<div class="cry-stats-area" id="cryStatsArea">
    <div class="cry-stats-title">🔴 哭声统计</div>
    <div class="cry-stats-empty" id="cryStatsEmpty">暂无哭声事件</div>
</div>

<button class="btn btn-start" id="startBtn" onclick="startStream()">开始传输</button>
<button class="btn btn-stop" id="stopBtn" onclick="stopStream()" disabled>停止传输</button>

<div class="log-area" id="logArea"><div class="log-entry">等待开始传输...</div></div>

<script>
let pc = null, localStream = null, startTime = null, durationTimer = null;
let evtSource = null;

const $ = id => document.getElementById(id);

function setStatus(state, text) {
    $('dot').className = 'dot ' + state;
    $('statusText').textContent = text || state;
}

function addLog(msg, cls) {
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (cls ? ' ' + cls : '');
    entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
    $('logArea').appendChild(entry);
    $('logArea').scrollTop = $('logArea').scrollHeight;
    while ($('logArea').children.length > 60) $('logArea').removeChild($('logArea').firstChild);
    // 将哭声（红色日志）归总到哭声统计栏
    if (cls === 'cry') {
        const empty = $('cryStatsEmpty');
        if (empty) empty.style.display = 'none';
        const cryEntry = document.createElement('div');
        cryEntry.className = 'cry-entry';
        const ts = new Date().toLocaleTimeString();
        cryEntry.innerHTML = '<span class="time">[' + ts + ']</span> ' + msg;
        $('cryStatsArea').appendChild(cryEntry);
        $('cryStatsArea').scrollTop = $('cryStatsArea').scrollHeight;
        // 保留最近 50 条哭声统计
        const cryEntries = $('cryStatsArea').querySelectorAll('.cry-entry');
        while (cryEntries.length > 50) cryEntries[0].remove();
    }
}

function showAlert(msg) {
    const box = $('alertBox');
    box.textContent = msg;
    box.style.display = 'block';
    setTimeout(() => { box.style.display = 'none'; }, 4000);
}

// SSE：接收电脑端的实时检测结果
function connectSSE() {
    evtSource = new EventSource('/events');
    evtSource.onopen = function() {
        addLog('SSE 连接已建立', 'info');
    };
    evtSource.onerror = function() {
        addLog('SSE 连接错误，浏览器将自动重连...', 'cry');
    };
    evtSource.onmessage = function(e) {
        const data = JSON.parse(e.data);
        // 始终同步推理次数、哭声事件、哭声评分（不受 cry_detected 影响）
        if (data.inference_count !== undefined) $('infCount').textContent = data.inference_count;
        if (data.cry_events !== undefined) $('cryCount').textContent = data.cry_events;
        if (data.max_cry_score !== undefined) $('cryScore').textContent = data.max_cry_score.toFixed(4);
        if (data.cry_detected) {
            setStatus('cry', '检测到哭声！');
            showAlert('*** 检测到哭声 ***  得分: ' + data.max_cry_score.toFixed(4));
            addLog('*** 检测到哭声 *** 得分: ' + data.max_cry_score.toFixed(4), 'cry');
            if (data.cry_details) {
                data.cry_details.forEach(d => addLog('  ' + d.class + ' (' + d.score.toFixed(4) + ')', 'cry'));
            }
        } else if (data.top_classes && data.top_classes.length > 0) {
            const top = data.top_classes.slice(0, 3).map(c => c.name + '(' + c.score.toFixed(2) + ')').join(', ');
            addLog('Top: ' + top, 'info');
        }
    };
}

function updateDuration() {
    if (startTime) $('duration').textContent = Math.floor((Date.now() - startTime) / 1000) + 's';
}

async function startStream() {
    try {
        setStatus('connecting', '连接中...');
        $('connState').textContent = '获取麦克风...';
        $('startBtn').disabled = true;
        addLog('正在请求麦克风权限...');

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('浏览器不支持麦克风访问。请使用 https:// 地址访问本页面。');
        }

        localStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        const track = localStream.getAudioTracks()[0];
        const settings = track.getSettings();
        $('audioRate').textContent = (settings.sampleRate || '?') + ' Hz';
        addLog('麦克风已授权, 采样率: ' + (settings.sampleRate || 'unknown') + ' Hz');

        // 建立 WebRTC 连接
        $('connState').textContent = '建立 WebRTC...';
        // 使用多个 STUN 服务器避免单点故障：
        //   - Google STUN 可能在国内不可达导致 40s 超时
        //   - 腾讯 STUN 作为国内备选
        //   - 对于 Tailscale/LAN 连接，host candidates 已足够，STUN 非必需
        const localPc = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun.qq.com:3478' },
            ],
            // 可选：使用 relay 模式强制走 TURN，但默认 all 即可
            iceTransportPolicy: 'all',
        });
        pc = localPc;  // 更新全局引用（供 stopStream 关闭用）

        localPc.onconnectionstatechange = () => {
            $('connState').textContent = localPc.connectionState;
            if (localPc.connectionState === 'connected') {
                setStatus('connected', '已连接');
                startTime = Date.now();
                durationTimer = setInterval(updateDuration, 1000);
                addLog('WebRTC 连接已建立');
            } else if (['failed', 'disconnected', 'closed'].includes(localPc.connectionState)) {
                setStatus('error', '连接断开');
                addLog('连接断开: ' + localPc.connectionState, 'cry');
            }
        };

        localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
        addLog('音频轨道已添加');

        // 创建 Offer 并等待 ICE 候选收集（带超时，避免 STUN 不可达时阻塞 40 秒）
        // 原因：stun.l.google.com 在国内可能不可达，STUN 超时约 40 秒。
        // 而 Tailscale/LAN 环境下 host candidates 已足够建立直连。
        const offer = await localPc.createOffer();
        await localPc.setLocalDescription(offer);

        // 等待 ICE 候选收集完成，最多等待 8 秒
        const ICE_GATHERING_TIMEOUT_MS = 8000;
        await Promise.race([
            new Promise(resolve => {
                if (localPc.iceGatheringState === 'complete') return resolve();
                localPc.addEventListener('icegatheringstatechange', () => {
                    if (localPc.iceGatheringState === 'complete') resolve();
                });
            }),
            new Promise(resolve => setTimeout(() => {
                addLog('ICE 候选收集超时 (' + ICE_GATHERING_TIMEOUT_MS / 1000 + 's)，使用已收集的候选继续');
                resolve();
            }, ICE_GATHERING_TIMEOUT_MS))
        ]);
        addLog('ICE 候选收集完成 (' + localPc.iceGatheringState + ')');

        // 发送 Offer 到服务器，接收 Answer
        const resp = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: localPc.localDescription.sdp, type: localPc.localDescription.type })
        });
        if (!resp.ok) throw new Error('服务器返回 ' + resp.status);
        const answer = await resp.json();
        await localPc.setRemoteDescription(new RTCSessionDescription(answer));
        addLog('信令交换完成');

        // 连接 SSE 接收实时结果
        connectSSE();

        $('stopBtn').disabled = false;
    } catch (err) {
        setStatus('error', '失败');
        $('connState').textContent = '失败';
        $('startBtn').disabled = false;
        addLog('错误: ' + err.message, 'cry');
        console.error(err);
        cleanup();
    }
}

function cleanup() {
    if (durationTimer) { clearInterval(durationTimer); durationTimer = null; }
    if (evtSource) { evtSource.close(); evtSource = null; }
    if (pc) { pc.close(); pc = null; }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
}

function stopStream() {
    addLog('正在停止传输...');
    cleanup();
    setStatus('idle', '已停止');
    $('connState').textContent = '已断开';
    $('audioRate').textContent = '--';
    $('startBtn').disabled = false;
    $('stopBtn').disabled = true;
    startTime = null;
    addLog('传输已停止');
    // 重置哭声统计栏
    $('cryStatsArea').querySelectorAll('.cry-entry').forEach(e => e.remove());
    $('cryStatsEmpty').style.display = '';
}
</script>
</body>
</html>"""


# ============================================================
# SSL 自签名证书生成
# ============================================================

def generate_ssl_cert() -> ssl.SSLContext | None:
    """生成自签名 SSL 证书（手机端 getUserMedia 要求 HTTPS 安全上下文）。

    优先用 openssl 命令，不可用时回退到 cryptography 库。
    返回 SSLContext 或 None（无法生成时回退到 HTTP）。
    """
    cert_dir = Path.home() / ".voice_transmission_server"
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists():
        logger.info(f"使用已有 SSL 证书: {cert_path}")
    else:
        cert_dir.mkdir(parents=True, exist_ok=True)
        # 方法 1: openssl
        ok = False
        try:
            subprocess.run(
                ["openssl", "req", "-x509", "-newkey", "rsa:2048",
                 "-keyout", str(key_path), "-out", str(cert_path),
                 "-days", "3650", "-nodes",
                 "-subj", "/CN=VoiceTransmissionServer"],
                capture_output=True, check=True, timeout=30,
            )
            ok = True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # 方法 2: cryptography 库
        if not ok:
            try:
                from cryptography import x509
                from cryptography.x509.oid import NameOID
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                import datetime

                key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                subject = issuer = x509.Name([
                    x509.NameAttribute(NameOID.COMMON_NAME, "VoiceTransmissionServer"),
                ])
                cert = (
                    x509.CertificateBuilder()
                    .subject_name(subject)
                    .issuer_name(issuer)
                    .public_key(key.public_key())
                    .serial_number(x509.random_serial_number())
                    .not_valid_before(datetime.datetime.utcnow())
                    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
                    .sign(key, hashes.SHA256())
                )
                key_path.write_bytes(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ))
                cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
                ok = True
            except ImportError:
                pass

        if ok:
            logger.info(f"自签名证书已生成: {cert_path}")
        else:
            logger.warning("无法生成 SSL 证书（openssl 和 cryptography 均不可用）。")
            logger.warning("手机端浏览器将无法获取麦克风权限。")
            return None

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(cert_path), str(key_path))
    return ssl_ctx


# ============================================================
# 网络工具：IP 检测与防火墙配置
# ============================================================

def detect_ips() -> tuple[str | None, list[str]]:
    """检测本机网络 IP，区分 Tailscale (100.64.0.0/10) 和局域网。

    Returns:
        (tailscale_ip, lan_ips)
    """
    all_ips = []

    # 方法 1: psutil 枚举网卡（最可靠，不触发 DNS）
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    all_ips.append(addr.address)
    except (ImportError, Exception):
        pass

    # 方法 2: UDP connect 探测出口 IP（查路由表，不发包）
    for probe in ("100.64.0.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(1)
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    all_ips.append(ip)
        except OSError:
            pass

    # 去重
    seen = set()
    unique = [ip for ip in all_ips if not (ip in seen or seen.add(ip))]

    # 分类
    tailscale_ip = None
    lan_ips = []
    for ip in unique:
        parts = ip.split(".")
        if len(parts) == 4 and ip.startswith("100.") and 64 <= int(parts[1]) <= 127:
            tailscale_ip = ip
        else:
            lan_ips.append(ip)
    return tailscale_ip, lan_ips


def configure_firewall(port: int):
    """在 Windows 防火墙添加入站规则（允许指定端口）。"""
    if sys.platform != "win32":
        return
    rule_name = f"VoiceTransmissionServer_Port{port}"
    try:
        check = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"],
            capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace",
        )
        if "No rules match" not in check.stdout and check.returncode == 0:
            logger.info(f"防火墙规则已存在: 端口 {port}")
            return
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "add", "rule",
             f"name={rule_name}", "dir=in", "action=allow",
             "protocol=TCP", f"localport={port}"],
            capture_output=True, timeout=5, encoding="utf-8", errors="replace",
        )
        logger.info(f"已添加防火墙入站规则: 端口 {port}")
    except Exception as e:
        logger.warning(f"防火墙配置失败 ({e})，请手动执行（管理员权限）:")
        logger.warning(f'  netsh advfirewall firewall add rule name="{rule_name}" dir=in action=allow protocol=TCP localport={port}')


# ============================================================
# WebRTC + HTTP 服务器
# ============================================================

class VoiceServer:
    """主服务器：HTTP 路由 + WebRTC 信令 + 音频处理。"""

    def __init__(self, host: str, port: int, threshold: float,
                 ssl_ctx: ssl.SSLContext | None,
                 teacher_url: str | None = None,
                 speech_detector=None,
                 enable_cry: bool = True):
        self.host = host
        self.port = port
        self.ssl_ctx = ssl_ctx
        self.app = web.Application()
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_post("/offer", self._handle_offer)
        self.app.router.add_get("/events", self._handle_sse)

        self.speech_detector = speech_detector
        self.enable_cry = enable_cry
        self.detector = (
            PannsCryDetector(threshold=threshold) if enable_cry else None
        )
        self.sse_manager = SSEManager()
        self.teacher_notifier = (
            TeacherNotifier(teacher_url) if teacher_url else None
        )
        self.pc: RTCPeerConnection | None = None
        self.processor: AudioProcessor | None = None
        self.audio_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # 主事件循环引用（供线程回调）

    # ----------------------------------------------------------
    # HTTP 路由
    # ----------------------------------------------------------

    async def _handle_index(self, _request):
        return web.Response(content_type="text/html", text=HTML_PAGE)

    async def _handle_offer(self, request):
        """处理 WebRTC Offer，返回 Answer。"""
        try:
            data = await request.json()
            logger.info(f"收到 WebRTC Offer from {request.remote}")

            # 关闭旧连接
            await self._close_connection()

            self.pc = RTCPeerConnection()

            @self.pc.on("track")
            async def on_track(track):
                logger.info(f"收到轨道: {track.kind}")
                if track.kind == "audio":
                    logger.info("音频轨道已就绪，开始创建 AudioProcessor 并启动处理循环...")
                    self.processor = AudioProcessor(
                        self.detector,
                        on_result=self._on_inference_result,
                        speech_detector=self.speech_detector,
                    )
                    self.audio_task = asyncio.create_task(self._process_audio(track))

            @self.pc.on("connectionstatechange")
            async def on_state_change():
                state = self.pc.connectionState
                logger.info(f"连接状态: {state}")
                if state in ("failed", "disconnected", "closed"):
                    logger.warning(f"WebRTC 连接 {state}")

            # 设置远程描述并创建 Answer
            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            logger.info("WebRTC Answer 已创建")
            return web.json_response({
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type,
            })
        except Exception as e:
            logger.error(f"处理 Offer 出错: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_sse(self, request):
        """SSE 端点：向手机端推送实时检测结果。"""
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        resp.headers["X-Accel-Buffering"] = "no"  # 防止反向代理缓冲 SSE 数据
        await resp.prepare(request)
        await self.sse_manager.add_client(resp)
        try:
            # 立即发送初始消息确认连接，并同步计数器初始值
            if self.processor:
                await resp.write(
                    f"data: {json.dumps({'cry_detected': False, 'max_cry_score': self.processor.last_max_cry_score, 'cry_events': self.processor.cry_events, 'inference_count': self.processor.inference_count, 'elapsed': 0, 'top_classes': [], 'cry_details': []}, ensure_ascii=False)}\n\n".encode("utf-8")
                )
            else:
                await resp.write(b"data: {}\n\n")
            await resp.drain()
            # 保持连接，直到客户端断开
            while True:
                await asyncio.sleep(5)  # 5 秒一次心跳，防止 NAT/代理超时断开空闲连接
                transport = request.transport
                if transport is None or transport.is_closing():
                    break
                await resp.write(b": heartbeat\n\n")
                await resp.drain()
        except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            await self.sse_manager.remove_client(resp)
        return resp

    # ----------------------------------------------------------
    # 音频处理循环
    # ----------------------------------------------------------

    async def _process_audio(self, track):
        """持续从音频轨道接收帧并推理。"""
        logger.info("音频处理循环已启动")
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                has_enough = self.processor.add_frame(frame)
                if has_enough:
                    # 在独立线程执行推理，避免阻塞事件循环
                    await asyncio.to_thread(self.processor.process)
        except asyncio.CancelledError:
            logger.info("音频处理任务已取消")
        except Exception as e:
            logger.error(f"音频处理出错: {e}", exc_info=True)
        finally:
            logger.info("音频处理循环已结束")

    def _on_inference_result(self, result: dict):
        """推理结果回调：通过 SSE 广播到手机端，并通知教师端（哭声事件）。

        此方法在 to_thread 的工作线程中调用，需通过 run_coroutine_threadsafe
        将广播任务调度到主事件循环。
        """
        # 构造精简的 SSE 消息（避免传输大数组）
        sse_data = {
            "cry_detected": result["cry_detected"],
            "max_cry_score": round(result["max_cry_score"], 4),
            "cry_events": result.get("cry_events", 0),
            "inference_count": result.get("inference_count", 0),
            "elapsed": result.get("elapsed"),
            "top_classes": result.get("top_classes", [])[:5],
            "cry_details": result.get("cry_details", []),
        }
        # 每 10 次推理打印一次 SSE 推送确认
        inf_count = sse_data["inference_count"]
        if inf_count == 1 or inf_count % 10 == 0:
            logger.info(
                f"SSE 推送 #{inf_count}: cry_detected={sse_data['cry_detected']}, "
                f"cry_events={sse_data['cry_events']}, "
                f"max_cry_score={sse_data['max_cry_score']:.4f}"
            )
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.sse_manager.broadcast(sse_data), self._loop
            )
        else:
            logger.warning(
                f"无法推送 SSE: _loop 未就绪或已停止 "
                f"(_loop={self._loop is not None})"
            )

        # 如果检测到哭声，通知教师端
        if result.get("cry_detected") and self.teacher_notifier:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.teacher_notifier.send_cry_alert(), self._loop
                )
            else:
                logger.warning(
                    "无法通知教师端: 事件循环未就绪"
                )

    # ----------------------------------------------------------
    # 连接管理
    # ----------------------------------------------------------

    async def _close_connection(self):
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
            self.audio_task = None
        if self.pc:
            await self.pc.close()
            self.pc = None
        self.processor = None

    # ----------------------------------------------------------
    # 启动
    # ----------------------------------------------------------

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, ssl_context=self.ssl_ctx)
        await site.start()

        # 保存主事件循环引用，供工作线程中的回调使用
        self._loop = asyncio.get_running_loop()

        tailscale_ip, lan_ips = detect_ips()
        configure_firewall(self.port)

        protocol = "https" if self.ssl_ctx else "http"
        print("\n" + "=" * 60)
        print("  麦克风流式传输与哭声识别服务器（PANNs）")
        print("=" * 60)
        print(f"\n  本地访问:    {protocol}://127.0.0.1:{self.port}")
        if tailscale_ip:
            print(f"  Tailscale:   {protocol}://{tailscale_ip}:{self.port}  <-- 手机用此地址")
        for ip in lan_ips:
            print(f"  局域网:      {protocol}://{ip}:{self.port}")

        if not tailscale_ip:
            print(f"\n  ⚠ 未检测到 Tailscale IP，请确认 Tailscale 已启动。")
            print(f"    运行 'ipconfig' 查看 100.x.x.x 地址。")

        if self.ssl_ctx:
            print(f"\n  🔒 HTTPS 已启用（自签名证书）")
            print(f"  📱 首次访问时请在浏览器中接受证书警告")
        else:
            print(f"\n  ⚠ 未启用 HTTPS，手机浏览器无法获取麦克风权限！")
            print(f"    请安装 openssl 或 pip install cryptography")

        if self.detector is not None:
            print(f"\n  哭声检测阈值: {self.detector.threshold}")
            print(f"  使用设备: {self.detector.device}")
            print(f"  监控的哭声类别:")
            for cid, name in self.detector.cry_classes.items():
                print(f"    [{cid}] {name}")
        else:
            print("\n  哭声检测: 已关闭（仅传输 WAIT_CARD 语音请求）")
        print(f"\n  按 Ctrl+C 停止服务器。")
        print("=" * 60 + "\n")

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        logger.info("正在关闭服务器...")
        await self._close_connection()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="麦克风流式传输与哭声识别服务器（PANNs）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python voice_transmission_server_panns.py
  python voice_transmission_server_panns.py --port 9000 --threshold 0.25
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CRY_THRESHOLD,
                        help=f"哭声检测阈值 0~1 (默认: {DEFAULT_CRY_THRESHOLD})")
    parser.add_argument("--teacher-url", type=str, default=DEFAULT_TEACHER_URL,
                        help=f"教师端 WebSocket 地址 (默认: {DEFAULT_TEACHER_URL})")
    args = parser.parse_args()

    # 依赖检查
    missing = []
    for mod in ("aiohttp", "aiortc", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    for mod in ("torch", "panns_inference"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"安装: pip install {' '.join(missing)}")
        sys.exit(1)

    if not HAS_SCIPY:
        print("[提示] 未安装 scipy，将使用 numpy 线性插值重采样。")
        print("       更高质量: pip install scipy")

    # 生成 SSL 证书
    ssl_ctx = generate_ssl_cert()

    # 启动服务器
    server = VoiceServer(
        host=args.host, port=args.port,
        threshold=args.threshold, ssl_ctx=ssl_ctx,
        teacher_url=args.teacher_url,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n正在关闭...")
        asyncio.run(server.shutdown())


if __name__ == "__main__":
    main()
