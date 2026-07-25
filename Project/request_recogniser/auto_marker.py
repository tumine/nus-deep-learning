from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# 1. 定义你的文本提示词与对应的类别名称
ontology = CaptionOntology({
    "pen": "pen",
    "eraser": "eraser",
    "toy building block": "building_block"
})

# 2. 加载预训练的 Grounding DINO 自动标注器
base_model = GroundingDINO(ontology=ontology)

# 3. 传入无标注图片目录，自动生成 YOLO 格式的数据集（包含 images, labels 和 data.yaml）
dataset = base_model.label(
    input_folder="Project/request_recogniser/collected_images",
    output_folder="Project/request_recogniser/yolo_dataset"
)

print("自动标注完成！可以直接用于 YOLOv11 训练。")
