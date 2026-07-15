"""
模型推理引擎
============

提供统一的推理接口，支持多种部署后端：

  - PyTorch (默认): 直接从 .pth 检查点加载，最灵活
  - TorchScript: 从 .pt 文件加载，跨平台，性能更好
  - ONNX Runtime: 从 .onnx 文件加载，CPU/GPU 通用，业界标准

设计原则：
  1. 模型只加载一次（懒加载 + 缓存）
  2. 预处理参数与训练时严格一致
  3. 线程安全（推理过程加锁）
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ============================================================
# 品种信息（与训练时一致）
# ============================================================

CAT_BREEDS = ["ragdoll", "singapura", "persian", "sphynx", "pallas"]

BREED_CN = {
    "ragdoll": "布偶猫",
    "singapura": "新加坡猫",
    "persian": "波斯猫",
    "sphynx": "斯芬克斯猫",
    "pallas": "兔狲",
}


class InferenceResult:
    """单次推理结果。"""

    def __init__(
        self,
        class_id: int,
        class_name: str,
        class_name_cn: str,
        confidence: float,
        top5_probs: list[dict],
        latency_ms: float,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.class_name_cn = class_name_cn
        self.confidence = confidence
        self.top5_probs = top5_probs  # [{rank, name, name_cn, probability}]
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_name_cn": self.class_name_cn,
            "confidence": round(self.confidence, 4),
            "top5": self.top5_probs,
            "latency_ms": round(self.latency_ms, 2),
        }


class CatBreedClassifier:
    """猫品种分类器 — 生产级推理封装。

    支持三种推理后端：

    - ``"pytorch"``: 从 .pth 检查点加载完整的 ResNet-50 模型
    - ``"torchscript"``: 从 .pt 文件加载 TorchScript 跟踪模型
    - ``"onnx"``: 通过 ONNX Runtime 加载 .onnx 模型

    Usage::

        # 方式 1: PyTorch 后端
        clf = CatBreedClassifier("best_model.pth", backend="pytorch")
        result = clf.predict("cat.jpg")

        # 方式 2: ONNX Runtime 后端（CPU 友好）
        clf = CatBreedClassifier("resnet50_cat.onnx", backend="onnx")
        result = clf.predict("cat.jpg")

        # 方式 3: 从 bytes 直接预测（适合 Web API）
        clf = CatBreedClassifier("resnet50_cat_scripted.pt", backend="torchscript")
        result = clf.predict_from_bytes(image_bytes)

        # 批量推理
        results = clf.predict_batch(["cat1.jpg", "cat2.jpg", "cat3.jpg"])
    """

    # ImageNet 标准化参数（必须与训练时一致）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 224

    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        device: str = "auto",
        use_fp16: bool = False,
    ):
        """
        Args:
            model_path: 模型文件路径 (.pth / .pt / .onnx)
            backend: 推理后端，可选 "pytorch" / "torchscript" / "onnx"
            device: 计算设备，"auto" 自动选择，"cpu" 或 "cuda"
            use_fp16: 是否启用 FP16（仅 CUDA 设备有效）
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.backend = backend.lower()
        if self.backend not in ("pytorch", "torchscript", "onnx"):
            raise ValueError(f"不支持的 backend: {backend}，可选 pytorch/torchscript/onnx")

        self.use_fp16 = use_fp16
        self._device = self._resolve_device(device)
        self._model: Optional[nn.Module] = None
        self._onnx_session = None
        self._lock = threading.Lock()  # 推理线程安全

        # 构建预处理 pipeline（与训练时一致）
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD,
            ),
        ])

    # ----------------------------------------------------------
    # 设备管理
    # ----------------------------------------------------------

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"自动选择 GPU: {gpu_name} ({vram:.1f} GB)")
                return torch.device("cuda:0")
            else:
                logger.info("CUDA 不可用，使用 CPU")
                return torch.device("cpu")
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    # ----------------------------------------------------------
    # 模型加载（懒加载 + 缓存）
    # ----------------------------------------------------------

    def load_model(self):
        """加载模型（如果已加载则跳过）。"""
        if self._model is not None or self._onnx_session is not None:
            return

        with self._lock:
            if self._model is not None or self._onnx_session is not None:
                return

            logger.info(f"加载模型 [{self.backend}]: {self.model_path}")

            if self.backend == "pytorch":
                self._load_pytorch()
            elif self.backend == "torchscript":
                self._load_torchscript()
            elif self.backend == "onnx":
                self._load_onnx()

    def _load_pytorch(self):
        """从 .pth 检查点加载 PyTorch 模型。"""
        # 构建与训练时相同的模型结构
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 5),
        )

        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

        model = model.to(self._device)
        if self.use_fp16 and self._device.type == "cuda":
            model = model.half()
        model.eval()

        self._model = model
        logger.info(f"PyTorch 模型加载完成 (参数: {sum(p.numel() for p in model.parameters()):,})")

    def _load_torchscript(self):
        """从 .pt 文件加载 TorchScript 模型。"""
        model = torch.jit.load(str(self.model_path), map_location=self._device)
        if self.use_fp16 and self._device.type == "cuda":
            model = model.half()
        model.eval()
        self._model = model
        logger.info("TorchScript 模型加载完成")

    def _load_onnx(self):
        """通过 ONNX Runtime 加载 .onnx 模型。"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX 后端需要 onnxruntime 包。安装: pip install onnxruntime-gpu"
            )

        # 检查可用的执行提供者
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime 可用提供者: {providers}")

        # 优先使用 CUDA，否则回退到 CPU
        if "CUDAExecutionProvider" in providers and self._device.type == "cuda":
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._onnx_session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("ONNX Runtime 使用 CUDA 后端")
        else:
            self._onnx_session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX Runtime 使用 CPU 后端")

    # ----------------------------------------------------------
    # 预处理
    # ----------------------------------------------------------

    def _preprocess(self, image) -> torch.Tensor:
        """预处理图片为模型输入 tensor。

        Args:
            image: PIL.Image 对象、numpy array (H,W,3) 或图片文件路径

        Returns:
            形状为 (1, 3, 224, 224) 的 tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的类型: {type(image)}，需要 PIL.Image / numpy array / str")

        tensor = self._transform(image).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor

    # ----------------------------------------------------------
    # 推理
    # ----------------------------------------------------------

    def predict(self, image) -> InferenceResult:
        """对单张图片进行分类预测。

        Args:
            image: PIL.Image 对象、numpy array (H,W,3) 或图片文件路径

        Returns:
            InferenceResult 对象
        """
        self.load_model()

        input_tensor = self._preprocess(image)

        with self._lock:
            with torch.no_grad():
                if self.backend == "onnx":
                    return self._predict_onnx(input_tensor)
                else:
                    return self._predict_torch(input_tensor)

    def predict_from_bytes(self, image_bytes: bytes) -> InferenceResult:
        """从图片字节数据直接预测（适合 Web API）。"""
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict(image)

    def predict_batch(
        self,
        images: list,
        batch_size: int = 8,
    ) -> list[InferenceResult]:
        """批量推理。

        Args:
            images: 图片路径、PIL.Image 或 numpy array 列表
            batch_size: 批次大小

        Returns:
            与输入顺序一致的结果列表
        """
        self.load_model()

        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # 预处理
            tensors = [self._preprocess(img) for img in batch]
            batch_tensor = torch.cat(tensors, dim=0)

            with self._lock:
                with torch.no_grad():
                    if self.backend == "onnx":
                        batch_results = self._predict_onnx_batch(batch_tensor)
                    else:
                        batch_results = self._predict_torch_batch(batch_tensor)

            results.extend(batch_results)

        return results

    # ----------------------------------------------------------
    # 内部推理实现
    # ----------------------------------------------------------

    def _predict_torch(self, input_tensor: torch.Tensor) -> InferenceResult:
        """PyTorch/TorchScript 单张推理。"""
        input_tensor = input_tensor.to(self._device)
        if self.use_fp16 and self._device.type == "cuda":
            input_tensor = input_tensor.half()

        t0 = time.perf_counter()
        outputs = self._model(input_tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return self._build_result(probs, elapsed)

    def _predict_torch_batch(self, input_tensor: torch.Tensor) -> list[InferenceResult]:
        """PyTorch/TorchScript 批量推理。"""
        input_tensor = input_tensor.to(self._device)
        if self.use_fp16 and self._device.type == "cuda":
            input_tensor = input_tensor.half()

        t0 = time.perf_counter()
        outputs = self._model(input_tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        batch_size = probs.shape[0]
        per_image_latency = elapsed / batch_size

        return [self._build_result(probs[i], per_image_latency) for i in range(batch_size)]

    def _predict_onnx(self, input_tensor: torch.Tensor) -> InferenceResult:
        """ONNX Runtime 单张推理。"""
        ort_inputs = {self._onnx_session.get_inputs()[0].name: input_tensor.numpy()}

        t0 = time.perf_counter()
        outputs = self._onnx_session.run(None, ort_inputs)
        elapsed = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]
        probs = self._softmax(logits)
        return self._build_result(probs, elapsed)

    def _predict_onnx_batch(self, input_tensor: torch.Tensor) -> list[InferenceResult]:
        """ONNX Runtime 批量推理。"""
        ort_inputs = {self._onnx_session.get_inputs()[0].name: input_tensor.numpy()}

        t0 = time.perf_counter()
        outputs = self._onnx_session.run(None, ort_inputs)
        elapsed = (time.perf_counter() - t0) * 1000

        logits = outputs[0]
        batch_size = logits.shape[0]
        probs = np.array([self._softmax(logits[i]) for i in range(batch_size)])
        per_image_latency = elapsed / batch_size

        return [self._build_result(probs[i], per_image_latency) for i in range(batch_size)]

    # ----------------------------------------------------------
    # 结果构建
    # ----------------------------------------------------------

    def _build_result(self, probs: np.ndarray, latency_ms: float) -> InferenceResult:
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [
            {
                "rank": rank + 1,
                "class_name": CAT_BREEDS[idx],
                "class_name_cn": BREED_CN.get(CAT_BREEDS[idx], CAT_BREEDS[idx]),
                "probability": round(float(probs[idx]), 4),
            }
            for rank, idx in enumerate(top5_idx)
        ]

        return InferenceResult(
            class_id=pred_idx,
            class_name=CAT_BREEDS[pred_idx],
            class_name_cn=BREED_CN.get(CAT_BREEDS[pred_idx], CAT_BREEDS[pred_idx]),
            confidence=confidence,
            top5_probs=top5,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# ============================================================
# 便捷工厂函数
# ============================================================

_MODEL_CACHE: dict[str, CatBreedClassifier] = {}
_CACHE_LOCK = threading.Lock()


def get_classifier(
    model_path: str,
    backend: str = "pytorch",
) -> CatBreedClassifier:
    """获取缓存的分类器实例（避免重复加载模型）。

    相同 model_path 对应同一个实例，适合在 Web 服务中使用。
    """
    cache_key = f"{model_path}:{backend}"
    if cache_key not in _MODEL_CACHE:
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                _MODEL_CACHE[cache_key] = CatBreedClassifier(
                    model_path=model_path,
                    backend=backend,
                )
    return _MODEL_CACHE[cache_key]
