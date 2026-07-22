# 课堂助手机器人 - ArUco 请求卡模块

## 项目概述

本模块是**课堂助手机器人**项目的一部分。

它负责识别学生在机器人停靠到其课桌后展示的 **ArUco 请求卡**。一旦请求卡被确认，该模块将生成配送任务并发送至机器人控制系统。

当前支持的课堂请求包括：

| 标记 ID | 请求内容 |
|---------|----------|
| 0 | 积木 |
| 1 | 铅笔 |
| 2 | 橡皮 |
| 3 | 老师帮助 |

---

## 工作流程

```
摄像头
    │
    ▼
ArUco 检测
    │
    ▼
请求映射
    │
    ▼
任务生成
    │
    ▼
任务队列
    │
    ▼
机器人控制器
```

机器人的工作流程如下：

1. 沿预设路线巡逻。
2. 检测举手的学生（由另一个模块实现）。
3. 停靠到学生旁边。
4. 等待学生展示 ArUco 请求卡。
5. 识别请求。
6. 生成配送任务。
7. 返回教师站领取所需物品。
8. 将物品递送给学生。
9. 恢复巡逻。

---

## 项目结构

```
card_detector/

│── main.py                 # 主程序
│── camera.py               # 摄像头接口
│── card_detector.py        # ArUco 检测模块
│── request_mapping.py      # 标记 ID → 课堂请求
│── request_manager.py      # 请求 → 配送任务
│── task_queue.py           # 机器人任务队列
│── robot_controller.py     # 机器人任务执行接口
│── state_machine.py        # 机器人状态管理器（待集成）
│── config.py               # 配置参数
```

---

## 当前功能

- ArUco 标记检测
- 多帧确认机制
- 课堂请求映射
- 配送任务生成
- 任务队列管理
- 机器人控制器接口
- 模块化架构，便于后续集成

---

## 未来集成计划

本模块将集成以下功能：

### YOLO 姿态检测

在巡逻过程中检测举手的学生。

### 机器人导航

控制机器人执行以下操作：

- 返回教师处
- 领取所需材料
- 导航返回学生处
- 恢复巡逻

### 树莓派

摄像头视频流可以通过以下两种方式提供：

- 本地 USB 摄像头
- 树莓派视频推流

仅需更改摄像头来源即可切换。

---

## 消息传递接口

系统采用结构化的消息传递架构，在模块之间传递检测事件、任务和机器人指令。所有消息均为带明确定义键的 Python 字典（`dict`）。

---

### 1. 举手事件

当学生连续多帧举手时，由 `HandDetector.detect()` 生成。

| 键 | 类型 | 描述 |
|-----|------|------|
| `type` | `str` | 始终为 `"hand_raise"`，标识这是一个手部检测事件 |
| `target` | `tuple(int, int)` | 检测到的人的边界框中心像素坐标 `(x, y)` |
| `confidence` | `float` | YOLO 检测置信度分数（0.0 ~ 1.0） |
| `bbox` | `tuple(int, int, int, int)` | 人的边界框：`(x1, y1, x2, y2)`，其中 `(x1, y1)` 为左上角，`(x2, y2)` 为右下角 |
| `left_raised` | `bool` | 如果左手腕在左肩上方（超出阈值）则为 `True` |
| `right_raised` | `bool` | 如果右手腕在右肩上方（超出阈值）则为 `True` |

**多帧确认**：手必须连续举起 `CONFIRM_FRAMES`（默认：5）帧才会触发事件，以防止误检测。

**示例**：
```python
{
    "type": "hand_raise",
    "target": (320, 240),
    "confidence": 0.87,
    "bbox": (200, 100, 440, 400),
    "left_raised": False,
    "right_raised": True
}
```

---

### 2. 卡片检测结果

当 ArUco 标记被检测并确认后，由 `CardDetector.detect()` 生成。

| 键 | 类型 | 描述 |
|-----|------|------|
| `id` | `int` | ArUco 标记 ID（0 ~ 3） |
| `request` | `str` | 从标记 ID 映射的可读请求名称：`"blocks"`、`"pencil"`、`"eraser"` 或 `"teacher"` |
| `center` | `tuple(int, int)` | 检测到的标记的中心像素坐标 `(x, y)` |
| `corners` | `np.ndarray` | 标记的四个角点，形状为 `(4, 2)` 的 `int` 数组 |
| `count` | `int` | 该标记已连续检测到的帧数 |
| `confirmed` | `bool` | 当标记已被检测到 `CONFIRM_FRAMES` 帧时为 `True`，防止重复触发 |

**多帧确认**：标记必须连续检测到 `CONFIRM_FRAMES` 帧。一旦确认，在它离开画面之前不会再次触发。

**ArUco 标记映射**：
| 标记 ID | `request` 值 | 含义 |
|---------|--------------|------|
| 0 | `"blocks"` | 积木 |
| 1 | `"pencil"` | 铅笔 |
| 2 | `"eraser"` | 橡皮 |
| 3 | `"teacher"` | 老师帮助 |

**示例**：
```python
{
    "id": 0,
    "request": "blocks",
    "center": (400, 300),
    "corners": array([[380, 280], [420, 280], [420, 320], [380, 320]]),
    "count": 5,
    "confirmed": True
}
```

---

### 3. 配送任务

由 `RequestManager.create_task()` 从已确认的卡片检测结果生成，然后由状态机补充学生上下文信息。

| 键 | 类型 | 描述 |
|-----|------|------|
| `type` | `str` | 始终为 `"delivery"`，标识这是一个配送任务 |
| `item` | `str` | 请求的物品名称（与卡片结果中的 `request` 相同）：`"blocks"`、`"pencil"`、`"eraser"` 或 `"teacher"` |
| `marker_id` | `int` | 触发此任务的原始 ArUco 标记 ID |
| `target` | `tuple(int, int)` | ArUco 标记在图像坐标中的中心点 |
| `student_context` | `dict` | （由 `main.py` 添加）卡片事件发生时状态机上下文的副本，包含学生位置和导航信息（参见[状态机上下文](#6-状态机上下文)） |

**示例**：
```python
{
    "type": "delivery",
    "item": "blocks",
    "marker_id": 0,
    "target": (400, 300),
    "student_context": {
        "route_node": 1,
        "scan_direction": "left",
        "student_target": (320, 240),
        "student_confidence": 0.87,
        "approach_command": {
            "direction": "left",
            "target": (320, 240),
            "forward_seconds": 1.5
        }
    }
}
```

---

### 4. 机器人指令（TCP 消息）

由 `RobotController.execute()` 通过 TCP 套接字以 JSON 编码字符串的形式发送给树莓派机器人控制系统。共有四种指令类型，每种对应一个状态转换。

#### 4.1 `approach_student` — 导航到举手的学生

| 键 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 始终为 `"approach_student"` |
| `route_node` | `int` 或 `None` | 机器人当前所在的路线节点编号 |
| `scan_direction` | `str` 或 `None` | 机器人正在扫描的方向：`"front"`、`"left"` 或 `"right"` |
| `student_target` | `tuple(int, int)` 或 `None` | 需要帮助的学生的中心坐标 `(x, y)` |
| `approach` | `dict` 或 `None` | 靠近移动参数：`{"direction": str, "target": (x,y), "forward_seconds": float}` |

#### 4.2 `go_teacher` — 导航返回教师站

| 键 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 始终为 `"go_teacher"` |
| `task` | `dict` | 完整的[配送任务](#3-配送任务)，包含请求的物品和学生上下文 |
| `route_node` | `int` 或 `None` | 机器人当前所在的路线节点 |

#### 4.3 `return_student` — 领取材料后返回学生处

| 键 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 始终为 `"return_student"` |
| `task` | `dict` | 已保存的[配送任务](#3-配送任务)（从 `StateMachine.get_task()` 获取） |
| `route_node` | `int` 或 `None` | 当前路线节点 |
| `scan_direction` | `str` 或 `None` | 靠近学生的方向：`"front"`、`"left"` 或 `"right"` |
| `student_target` | `tuple(int, int)` 或 `None` | 学生的坐标 |
| `approach` | `dict` 或 `None` | 靠近移动参数 |

#### 4.4 `return_patrol` — 恢复巡逻路线

| 键 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 始终为 `"return_patrol"` |
| `route_node` | `int` 或 `None` | 要返回的路线节点 |
| `scan_direction` | `str` 或 `None` | 要恢复的原始扫描方向 |
| `approach` | `dict` 或 `None` | 供参考的靠近参数 |

**TCP 通信**：
- **协议**：TCP（流式）
- **主机**：`100.84.2.68`（可在 `RobotController` 中配置）
- **端口**：`2105`（可配置）
- **格式**：UTF-8 编码的 JSON 字符串
- **超时**：2.0 秒
- **流控**：`RobotController.busy` 标志位阻止在前一个指令仍在执行时发送新指令

**示例**（approach_student）：
```json
{
    "command": "approach_student",
    "route_node": 1,
    "scan_direction": "left",
    "student_target": [320, 240],
    "approach": {
        "direction": "left",
        "target": [320, 240],
        "forward_seconds": 1.5
    }
}
```

---

### 5. 人物信息（内部）

由 `HandDetector` 内部为每个检测到的人物生成，供 `HandDetector.draw()` 用于可视化。

| 键 | 类型 | 描述 |
|-----|------|------|
| `bbox` | `tuple(int, int, int, int)` | 人物边界框 `(x1, y1, x2, y2)` |
| `center` | `tuple(int, int)` | 边界框的中心点 `(x, y)` |
| `confidence` | `float` | YOLO 检测置信度（0.0 ~ 1.0） |
| `left_shoulder` | `tuple(int, int)` | COCO 关键点 5 — 左肩像素坐标 |
| `right_shoulder` | `tuple(int, int)` | COCO 关键点 6 — 右肩像素坐标 |
| `left_wrist` | `tuple(int, int)` | COCO 关键点 9 — 左手腕像素坐标 |
| `right_wrist` | `tuple(int, int)` | COCO 关键点 10 — 右手腕像素坐标 |
| `left_raised` | `bool` | 左手腕是否在左肩上方 |
| `right_raised` | `bool` | 右手腕是否在右肩上方 |
| `hand_raised` | `bool` | 任一只手举起则为 `True` |

---

### 6. 状态机上下文

由 `StateMachine` 在机器人整个工作流程中维护。以字典形式存储并在组件间传递。

| 键 | 类型 | 初始值 | 描述 |
|-----|------|--------|------|
| `route_node` | `int` 或 `None` | `None` | 机器人当前所在的路线/交叉节点 |
| `scan_direction` | `str` 或 `None` | `None` | 机器人正在扫描/面对的方向：`"front"`、`"left"` 或 `"right"` |
| `student_target` | `tuple` 或 `None` | `None` | 需要服务的学生的像素坐标 |
| `student_confidence` | `float` 或 `None` | `None` | 举手检测的置信度分数 |
| `approach_command` | `dict` 或 `None` | `None` | 靠近学生的移动参数：`{"direction": str, "target": (x,y), "forward_seconds": float}` |

**上下文方法**：
- `update_context(**kwargs)` — 更新一个或多个上下文键（对未知键会抛出 `KeyError`）
- `get_context()` — 返回完整上下文字典的浅拷贝
- `get_context_value(key, default)` — 获取单个值并带有回退默认值
- `clear_context()` — 将所有上下文值重置为 `None`

---

### 7. 标记状态（内部）

由 `CardDetector` 内部为每个检测到的 ArUco 标记维护。

| 键 | 类型 | 描述 |
|-----|------|------|
| `count` | `int` | 该标记已连续检测到的帧数 |
| `confirmed` | `bool` | 当标记已被看到 `CONFIRM_FRAMES` 帧后为 `True` |

---

### 消息流图

```
                                    机器人状态
                                       │
┌──────────┐   举手事件        ┌───────▼────────────┐
│HandDetect├──────────────────►│ APPROACH_STUDENT   │
│  .detect()│                  │   （靠近学生）       │
└──────────┘                   └───────┬────────────┘
                                       │
                            approach_student 指令
                                       │ (TCP JSON)
┌──────────┐   卡片结果       ┌───────▼────────┐     任务      ┌──────────┐
│CardDetect├─────────────────►│   WAIT_CARD    ├──────────────►│TaskQueue │
│  .detect()│                 │  （等待卡片）   │               │（任务队列）│
└──────────┘                  └───────┬────────┘               └────┬─────┘
                                       │                            │
                            go_teacher 指令                  next_task()
                                       │ (TCP JSON)                 │
                              ┌───────▼────────┐                   │
                              │   GO_TEACHER   ◄───────────────────┘
                              │  （前往教师处）  │
                              └───────┬────────┘
                                       │
                              ┌───────▼────────┐
                              │  WAIT_LOADING  │（教师装载物品）
                              │  （等待装载）   │
                              └───────┬────────┘
                                       │
                         return_student 指令
                                       │ (TCP JSON)
                              ┌───────▼────────┐
                              │ RETURN_STUDENT │
                              │ （返回学生处）   │
                              └───────┬────────┘
                                       │
                              ┌───────▼────────┐
                              │  WAIT_UNLOAD   │（学生取走物品）
                              │  （等待卸载）   │
                              └───────┬────────┘
                                       │
                           return_patrol 指令
                                       │ (TCP JSON)
                              ┌───────▼────────┐
                              │ RETURN_PATROL  │──► 返回巡逻
                              │ （返回巡逻）    │
                              └────────────────┘
```

---

## 依赖

- Python 3.10+
- OpenCV
- OpenCV ArUco 模块

安装方式：

```bash
pip install opencv-python
pip install opencv-contrib-python
```

---

## 使用方法

运行

```bash
python main.py
```

按 **Q** 键退出。

---

## 备注

本模块当前专注于基于 ArUco 的课堂请求识别。

机器人的运动控制、YOLO 举手检测和树莓派通信将在后续开发阶段集成。

---

## 版本

当前版本：v0.2

### 已完成

- 摄像头模块
- ArUco 检测器
- 请求映射
- 任务生成
- 任务队列
- 机器人控制器接口

### 开发中

- YOLO 举手检测
- 状态机
- 树莓派视频推流
- 机器人通信
- 材料配送工作流
