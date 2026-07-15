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

# 注意：这些列表的顺序必须与训练时 ImageFolder 的字母序一致！
# ImageFolder 按目录名排序: pallas, persian, ragdoll, singapura, sphynx
# 如果有 class_info 保存于 checkpoint，优先从 checkpoint 读取。
_DEFAULT_CAT_BREEDS = ["pallas", "persian", "ragdoll", "singapura", "sphynx"]

BREED_CN = {
    "pallas": "兔狲",
    "persian": "波斯猫",
    "ragdoll": "布偶猫",
    "singapura": "新加坡猫",
    "sphynx": "斯芬克斯猫",
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
        is_not_cat: bool = False,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.class_name_cn = class_name_cn
        self.confidence = confidence
        self.top5_probs = top5_probs  # [{rank, name, name_cn, probability}]
        self.latency_ms = latency_ms
        self.is_not_cat = is_not_cat  # 是否被判定为非猫图片

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_name_cn": self.class_name_cn,
            "confidence": round(self.confidence, 4),
            "is_not_cat": self.is_not_cat,
            "top5": self.top5_probs,
            "latency_ms": round(self.latency_ms, 2),
        }


class CatBreedClassifier:
    """猫品种分类器 — 生产级推理封装。

    支持三种推理后端：

    - ``"pytorch"``: 从 .pth 检查点加载完整模型（支持 ResNet-50 / ConvNeXt / EfficientNetV2）
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
    DEFAULT_INPUT_SIZE = 224  # 向后兼容，实际值从 checkpoint 读取

    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        device: str = "auto",
        use_fp16: bool = False,
        input_size: Optional[int] = None,
        confidence_threshold: float = 0.50,
    ):
        """
        Args:
            model_path: 模型文件路径 (.pth / .pt / .onnx)
            backend: 推理后端，可选 "pytorch" / "torchscript" / "onnx"
            device: 计算设备，"auto" 自动选择，"cpu" 或 "cuda"
            use_fp16: 是否启用 FP16（仅 CUDA 设备有效）
            input_size: 手动指定输入分辨率（None 则从 checkpoint 读取，默认 224）
            confidence_threshold: 置信度阈值（0-1），低于此值判定为"非猫"
                推荐值: 0.50 (默认), 调高更保守, 调低更宽松
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.backend = backend.lower()
        if self.backend not in ("pytorch", "torchscript", "onnx"):
            raise ValueError(f"不支持的 backend: {backend}，可选 pytorch/torchscript/onnx")

        self.use_fp16 = use_fp16
        self.confidence_threshold = confidence_threshold
        self._device = self._resolve_device(device)
        self._model: Optional[nn.Module] = None
        self._onnx_session = None
        self._lock = threading.Lock()  # 推理线程安全

        # 从 checkpoint 读取元数据（PyTorch 后端）
        self.input_size = input_size or self.DEFAULT_INPUT_SIZE
        self._backbone = "resnet50"  # 默认兼容旧模型
        self.class_names = list(_DEFAULT_CAT_BREEDS)  # 默认类别列表
        self._has_other_class = False  # 是否包含"other"拒识类
        if self.backend == "pytorch":
            try:
                ckpt = torch.load(self.model_path, map_location="cpu")
                self.input_size = ckpt.get("input_size", self.DEFAULT_INPUT_SIZE)
                self._backbone = ckpt.get("backbone", "resnet50")
                # 优先从 checkpoint 读取类别顺序（与 ImageFolder 字母序一致）
                class_info = ckpt.get("class_info")
                if class_info and "classes" in class_info:
                    self.class_names = class_info["classes"]
                # 检测是否包含"other"拒识类
                self._has_other_class = "other" in self.class_names
                logger.info(
                    f"Checkpoint 信息: backbone={self._backbone}, "
                    f"input_size={self.input_size}, classes={self.class_names}"
                )
            except Exception:
                logger.warning("无法读取 checkpoint 元数据，使用默认配置")

        # 计算真正猫品种的索引（排除 "other"）
        self._cat_indices: list[int] = [
            i for i, name in enumerate(self.class_names)
            if name != "other"
        ]

        # 构建预处理 pipeline（与训练时一致）
        self._transform = transforms.Compose([
            transforms.Resize(int(self.input_size * 1.15)),
            transforms.CenterCrop(self.input_size),
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

    @property
    def has_other_class(self) -> bool:
        """模型是否包含"other"拒识类别。"""
        return self._has_other_class

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
        """从 .pth 检查点加载 PyTorch 模型。

        支持三种骨干网络，自动从 checkpoint 读取 backbone 类型：
        - resnet50 (原 train_cnn.py)
        - convnext_tiny / convnext_small / convnext_base (train_cnn_v2.py)
        - efficientnet_v2_s / efficientnet_v2_m (train_cnn_v2.py)
        """
        # 重新读取 checkpoint（确保拿到最新的元数据）
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        backbone = checkpoint.get("backbone", self._backbone)
        self._backbone = backbone
        self.input_size = checkpoint.get("input_size", self.input_size)

        # 从 checkpoint 读取实际类别（已在 __init__ 中解析为 self.class_names）
        class_info = checkpoint.get("class_info")
        if class_info and "classes" in class_info:
            self.class_names = class_info["classes"]
            self._has_other_class = "other" in self.class_names
            self._cat_indices = [
                i for i, name in enumerate(self.class_names) if name != "other"
            ]
            logger.info(f"从 checkpoint 加载类别: {self.class_names}")
        num_classes = len(self.class_names)
        dropout_p = 0.5  # 默认值

        # ----------------------------------------------------------
        # 根据 backbone 构建对应模型结构
        # ----------------------------------------------------------
        if backbone.startswith("convnext"):
            # ConvNeXt 系列
            dropout_p = 0.6 if "base" in backbone else 0.5
            pretrained_weights = getattr(
                models, f"ConvNeXt_{backbone.split('_')[1].capitalize()}_Weights"
            ).IMAGENET1K_V1
            model = getattr(models, f"convnext_{backbone.split('_')[1]}")(
                weights=pretrained_weights
            )
            classifier_in = model.classifier[2].in_features
            head_attr = "classifier"

        elif backbone.startswith("efficientnet_v2"):
            # EfficientNetV2 系列
            dropout_p = 0.6 if backbone.endswith("_m") else 0.5
            pretrained_weights = getattr(
                models, f"EfficientNet_V2_{backbone.split('_')[2].upper()}_Weights"
            ).IMAGENET1K_V1
            model = getattr(models, f"efficientnet_v2_{backbone.split('_')[2]}")(
                weights=pretrained_weights
            )
            classifier_in = model.classifier[1].in_features
            head_attr = "classifier"

        else:
            # ResNet-50（默认，向后兼容旧版 train_cnn.py）
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            classifier_in = model.fc.in_features
            head_attr = "fc"
            dropout_p = 0.6

        # ---- 构建分类头（与 train_cnn_v2.py 完全一致） ----
        # 注意：ConvNeXt 的 classifier 内部包含 Flatten，替换后需补上
        new_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(classifier_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.8),
            nn.Linear(512, num_classes),
        )

        # 替换分类头
        setattr(model, head_attr, new_head)

        # 加载权重
        model.load_state_dict(state_dict, strict=True)

        model = model.to(self._device)
        if self.use_fp16 and self._device.type == "cuda":
            model = model.half()
        model.eval()

        self._model = model
        logger.info(
            f"PyTorch 模型加载完成 "
            f"(backbone={backbone}, input_size={self.input_size}, "
            f"参数: {sum(p.numel() for p in model.parameters()):,})"
        )

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
            形状为 (1, 3, H, W) 的 tensor，其中 H=W=self.input_size
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
        """构建推理结果，包含非猫拒识逻辑。

        由于训练集已移除 "other" 类别，仅使用置信度阈值兜底：
        - 若最高置信度低于阈值 → 判定为非猫图片
        """
        class_names = self.class_names
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [
            {
                "rank": rank + 1,
                "class_name": class_names[idx],
                "class_name_cn": BREED_CN.get(class_names[idx], class_names[idx]),
                "probability": round(float(probs[idx]), 4),
            }
            for rank, idx in enumerate(top5_idx)
        ]

        # 最高概率类别
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_name = class_names[pred_idx]

        # ---- 非猫拒识判断（仅阈值兜底）----
        is_not_cat = False

        if self._has_other_class and pred_name == "other":
            # 情形1: 模型有 other 类别且预测为 other → 明确拒识
            is_not_cat = True
            confidence = float(probs[pred_idx])
        elif confidence < self.confidence_threshold:
            # 情形2: 最高置信度不足阈值 → 兜底拒识
            is_not_cat = True

        # 统一处理拒识结果
        if is_not_cat:
            pred_name = "other"
            pred_idx = -1  # 非法索引，标记为拒识

        return InferenceResult(
            class_id=pred_idx,
            class_name=pred_name,
            class_name_cn=BREED_CN.get(pred_name, "非猫/其他"),
            confidence=confidence,
            top5_probs=top5,
            latency_ms=latency_ms,
            is_not_cat=is_not_cat,
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
