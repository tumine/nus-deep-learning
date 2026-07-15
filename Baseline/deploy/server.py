"""
猫品种识别 — FastAPI REST 推理服务
===================================

基于 FastAPI 的生产级推理 API，提供以下接口：

  - POST /predict        单张图片分类
  - POST /predict/batch  批量图片分类
  - GET  /health         健康检查
  - GET  /info           模型信息

启动方式：

    # 默认 PyTorch 后端
    python -m deploy.server --model outputs/.../best_model.pth

    # ONNX Runtime 后端（CPU 友好）
    python -m deploy.server --model outputs/.../resnet50_cat.onnx --backend onnx

    # 指定端口和工作进程数
    python -m deploy.server --model best_model.pth --port 8080 --workers 2

生产环境启动（推荐使用 uvicorn 直接启动）：

    uvicorn deploy.server:create_app --factory --host 0.0.0.0 --port 8000 --workers 2

API 文档自动生成：
    启动后访问 http://localhost:8000/docs 查看 Swagger UI
"""

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 将 deploy 父目录加入 sys.path，以便 import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from deploy.inference import BREED_CN, CAT_BREEDS, CatBreedClassifier, get_classifier

# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cat-breed-api")

# ============================================================
# 全局状态（通过 FastAPI lifespan 管理）
# ============================================================

_classifier: Optional[CatBreedClassifier] = None
_startup_time: Optional[float] = None
_model_config: dict = {}


# ============================================================
# Pydantic 数据模型
# ============================================================

class TopKItem(BaseModel):
    rank: int
    class_name: str
    class_name_cn: str
    probability: float


class PredictResponse(BaseModel):
    class_id: int = Field(..., description="预测类别索引 0-4")
    class_name: str = Field(..., description="品种英文名")
    class_name_cn: str = Field(..., description="品种中文名")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    top5: list[TopKItem] = Field(..., description="Top-5 概率分布")
    latency_ms: float = Field(..., description="推理耗时 (ms)")


class BatchPredictResponse(BaseModel):
    total: int = Field(..., description="图片总数")
    results: list[PredictResponse]
    total_latency_ms: float = Field(..., description="总推理耗时 (ms)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    device: str
    uptime_seconds: float
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理。"""
    global _classifier, _startup_time

    config = _model_config
    logger.info(f"正在加载模型: {config.get('model_path')}")
    logger.info(f"后端: {config.get('backend', 'pytorch')}")

    _classifier = get_classifier(
        model_path=config["model_path"],
        backend=config.get("backend", "pytorch"),
    )
    _classifier.load_model()
    _startup_time = time.time()

    logger.info("模型加载完成，服务就绪 ✅")
    yield

    logger.info("服务关闭")


def create_app(model_path: str = None, backend: str = "pytorch") -> FastAPI:
    """创建 FastAPI 应用（支持工厂模式，uvicorn --factory 使用）。"""
    if model_path:
        _model_config["model_path"] = model_path
        _model_config["backend"] = backend

    app = FastAPI(
        title="猫品种识别 API",
        description="基于 ResNet-50 的 5 种猫品种分类服务。支持单张/批量推理。",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 配置（允许跨域请求）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------
    # API 路由
    # ----------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="健康检查",
        description="返回服务运行状态和模型信息。Kubernetes 可用此端点做存活探测。",
    )
    async def health():
        if _classifier is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "detail": "模型未加载"},
            )

        uptime = time.time() - _startup_time if _startup_time else 0
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            backend=_classifier.backend,
            device=str(_classifier.device),
            uptime_seconds=round(uptime, 1),
        )

    @app.get(
        "/info",
        summary="模型信息",
        description="返回模型配置、支持类别等元信息。",
    )
    async def info():
        return {
            "model_type": "ResNet-50 (Fine-tuned)",
            "task": "Cat Breed Classification",
            "num_classes": len(CAT_BREEDS),
            "classes": [
                {"id": i, "name": CAT_BREEDS[i], "name_cn": BREED_CN.get(CAT_BREEDS[i], CAT_BREEDS[i])}
                for i in range(len(CAT_BREEDS))
            ],
            "input_size": "224x224 RGB",
            "backend": _classifier.backend if _classifier else "N/A",
            "device": str(_classifier.device) if _classifier else "N/A",
        }

    @app.post(
        "/predict",
        response_model=PredictResponse,
        summary="单张图片预测",
        description="上传一张猫的图片，返回品种分类结果（Top-5 概率分布）。",
        responses={
            200: {"description": "预测成功"},
            400: {"model": ErrorResponse, "description": "请求错误"},
            500: {"model": ErrorResponse, "description": "服务器内部错误"},
        },
    )
    async def predict(file: UploadFile = File(..., description="猫的图片文件 (jpg/png/webp)")):
        if _classifier is None:
            raise HTTPException(status_code=503, detail="模型尚未加载")

        # 校验文件类型
        content_type = file.content_type or ""
        if not any(t in content_type for t in ("image", "octet-stream")):
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {content_type}")

        try:
            image_bytes = await file.read()
            if len(image_bytes) == 0:
                raise HTTPException(status_code=400, detail="文件为空")

            result = _classifier.predict_from_bytes(image_bytes)
            return PredictResponse(**result.to_dict())

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")

    @app.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        summary="批量图片预测",
        description="上传多张猫图片（最多 20 张），返回批量分类结果。",
    )
    async def predict_batch(
        files: list[UploadFile] = File(..., description="猫图片文件列表 (最多 20 张)"),
    ):
        if _classifier is None:
            raise HTTPException(status_code=503, detail="模型尚未加载")

        if len(files) > 20:
            raise HTTPException(status_code=400, detail="批量推理最多支持 20 张图片")

        if len(files) == 0:
            raise HTTPException(status_code=400, detail="请至少上传一张图片")

        try:
            # 读取所有图片为 PIL Image
            images = []
            errors = []
            for file in files:
                try:
                    img_bytes = await file.read()
                    from PIL import Image
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    errors.append({"file": file.filename, "error": str(e)})

            if not images:
                raise HTTPException(status_code=400, detail=f"所有图片读取失败: {errors}")

            t0 = time.perf_counter()
            results = _classifier.predict_batch(images, batch_size=min(8, len(images)))
            total_latency = (time.perf_counter() - t0) * 1000

            response_data = []
            for i, r in enumerate(results):
                item = r.to_dict()
                if i < len(errors):
                    item["_warning"] = errors[i]
                response_data.append(item)

            return BatchPredictResponse(
                total=len(results),
                results=[PredictResponse(**d) for d in response_data],
                total_latency_ms=round(total_latency, 2),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"批量推理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")

    return app


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="猫品种识别 — FastAPI 推理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # PyTorch 后端
  python -m deploy.server --model outputs/resnet50_cat_xxx/checkpoints/best_model.pth

  # ONNX 后端 (CPU 友好)
  python -m deploy.server --model outputs/resnet50_cat_xxx/resnet50_cat.onnx --backend onnx

  # TorchScript 后端
  python -m deploy.server --model outputs/resnet50_cat_xxx/resnet50_cat_scripted.pt --backend torchscript
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (.pth/.pt/.onnx)")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "torchscript", "onnx"],
                        help="推理后端 (默认 pytorch)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--workers", type=int, default=1,
                        help="工作进程数 (仅 Linux/macOS 支持多进程)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    # 设置模型配置
    _model_config["model_path"] = args.model
    _model_config["backend"] = args.backend

    logger.info("=" * 60)
    logger.info("猫品种识别 API 服务")
    logger.info("=" * 60)
    logger.info(f"  模型:     {args.model}")
    logger.info(f"  后端:     {args.backend}")
    logger.info(f"  地址:     http://{args.host}:{args.port}")
    logger.info(f"  API 文档: http://{args.host}:{args.port}/docs")
    logger.info(f"  健康检查: http://{args.host}:{args.port}/health")
    logger.info("=" * 60)

    uvicorn.run(
        "deploy.server:create_app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
