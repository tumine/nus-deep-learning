"""
自动标注脚本 - 使用 Grounding DINO 为图片自动生成 YOLO 格式标注。

常见问题排查：
  1. SSL/HuggingFace 下载失败 → 设置环境变量 HF_ENDPOINT=https://hf-mirror.com
  2. torch.meshgrid 警告 → 已通过 warning filter 屏蔽，不影响运行
  3. 模型缓存位置 → ~/.cache/huggingface/hub/
"""

import os
import warnings

# ============================================================
# 修复 1: 屏蔽 torch.meshgrid 的 indexing 参数弃用警告
# ============================================================
warnings.filterwarnings(
    "ignore",
    message="torch\\.meshgrid: in an upcoming release.*",
    category=FutureWarning,
)

# ============================================================
# 修复 2: 设置 HuggingFace 镜像站以解决 SSL/网络连接问题
#   如果仍然下载失败，可以手动下载模型到本地缓存目录。
#   手动下载地址（将 bert-base-uncased 替换为 hf-mirror.com）：
#     https://hf-mirror.com/bert-base-uncased
#   模型缓存目录默认在：~/.cache/huggingface/hub/
# ============================================================
if not os.environ.get("HF_ENDPOINT"):
    # 使用国内镜像加速下载，解决 SSL UNEXPECTED_EOF_WHILE_READING 错误
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"已设置 HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# 1. 定义你的文本提示词与对应的类别名称
ontology = CaptionOntology({
    "pen": "pen",
    "eraser": "eraser",
    "toy building block": "building_block"
})

# 2. 加载预训练的 Grounding DINO 自动标注器
print("正在加载 Grounding DINO 模型（首次运行会下载 bert-base-uncased）...")
base_model = GroundingDINO(ontology=ontology)

# 3. 传入无标注图片目录，自动生成 YOLO 格式的数据集（包含 images, labels 和 data.yaml）
dataset = base_model.label(
    input_folder="Project/request_recogniser/collected_images",
    output_folder="Project/request_recogniser/yolo_dataset"
)

print("自动标注完成！可以直接用于 YOLOv11 训练。")
