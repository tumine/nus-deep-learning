## Date: 24 July, 2026

---

### 需求
根据当前小车的任务状态，通过手机麦克风播放对应的预录制回复。

### 交付要求
- 修改后的 `ws_car_control_V1_3.py`，实现在预录制语音播放完成后才进行语音播放。
- 一个 Python 程序 A，实现与本机上运行的 `main_speech.py` 的通信（传输语音文件并在网页上播放）。
- 修改后的 `main_speech.py`，在其运行过程中建立的网页上实现：接收到完整音频文件后，立刻播放。

### 部署位置
代码部署在笔记本电脑上。

### 硬件和软件拓扑结构
- 笔记本电脑上同时运行 `main_speech.py` 和 [交付要求] 中的新 Python 程序 A。
- 程序 A 提取保存在项目目录下的预录制语音文件，将它通过网络通信传输给 `main_speech.py`，在 `main_speech.py` 运行过程中建立的网页上播放。该网页由手机端打开，手机端同时作为麦克风与扬声器。
- `ws_car_control_V1_3.py` 在各个需要播放语音的时刻，调用程序 A 中的模块，实现在手机麦克风上播放语音的功能。
- 手机端与 `main_speech.py` 所在的笔记本电脑通过 Tailscale 网络连接。

### 参考程序
- integrated_pipeline.py，其中包含了借助 Tailscale 将树莓派摄像头画面截取到本机保存的程序。
- ws_car_avoid.py 是树莓派端的运行程序，通过网页对外开放视频流和运动控制 API，可在网页上点击截取视频页面。

### 程序实现详细要求
1. 对于 Python 程序 A，需要从它所在的笔记本电脑上的指定路径读取预录制的音频文件，并通过网络传输给 `main_speech.py`。
2. 对于 Python 程序 A，对外暴露 1 个调用接口参数，指示播放的音频编号。
3. 对于 `main_speech.py`，可以将接收到的完整音频（由 Python 程序 A 传输得到）在建立的网页上播放。
4. 对于 `ws_car_control_V1_3.py`，需要在以下场景下，分别调用一次 Python 程序 A（给定需要播放的音频序号）播放对应音频。在音频播放完成之前，不应“执行下一条运动指令”、“开始按钮按下检测”、或者“开始语音检测”。
   1. 小车走到学生面前（超声波传感器识别到障碍物并停下）之后。
   2. 学生通过 ArUco 码或者语音表述完需求（**请求物品、请求老师协助需要分别播放两种不同的音频**）之后。
   3. 小车到达老师一侧之后，等待老师放入物品时。
   4. 小车回到学生面前，等待学生取出物品时。
5. 对于音频播放任务，需要解决的问题：如何将语音播放完成的信号传递给 `ws_car_control_V1_3.py`？这个信号应该由 `main_speech.py` 给出还是 Python 程序 A 给出？

### 使用的音频内容
1. Please signify the ArUco code or speak out your request.
2. Your request has been transferred to the teacher. Please wait until I fetch you the item.
3. Your request has been transferred to the teacher. Please wait until I get the teacher here.
4. Please press the button after you've put the item in my basket.
5. Please claim your item from my basket, and press the button when you take it.

---
---

# 需求名称：基于小车任务状态的远程语音播放与动作同步控制系统

## 1. 系统架构与拓扑结构

* **运行环境：** 本地笔记本电脑（运行 Python 3.x，已建立 Tailscale 内网穿透网络）。
* **客户端/硬件：** 手机端（通过 Tailscale 连接笔记本，作为系统的前端网页载体、麦克风输入与扬声器输出）。
* **核心模块拓扑：**
  * `ws_car_control_V1_3.py`：小车主控状态机逻辑，决定何时播放何种语音。
  * `main_speech.py`：Web 服务器组件，负责维护与手机端 Web 页面之间的 HTTP/WebSocket 长连接，实现语音流传输与控制。
  * `Python 程序 A`：通信代理与音效调度模块，对外暴露简洁接口供 `ws_car_control_V1_3.py` 调用，对内与 `main_speech.py` 进行数据与信号交互。

---

## 2. 交付物清单

1. **修改后的 `ws_car_control_V1_3.py`：** 集成语音播放的阻塞式调用，保证“语音播完前不进行下一步操作”。
2. **新增 `Python 程序 A`（如 `audio_dispatcher.py`）：** 负责本地预录制音频文件的读取、传输以及与 `main_speech.py` 的进程间通信（IPC/RPC）。
3. **修改后的 `main_speech.py`：** 扩展网页服务端功能，支持实时接收音频推送并推送到前端；同时实现播放完成状态的回传。

---

## 3. 详细功能与接口要求

### 3.1 Python 程序 A 功能要求
1. **音频管理：** 维护本地预录制音频文件库（如按 ID 映射到路径 `./audio/1.m4a`, `./audio/2.m4a` 等）。
2. **对外接口：** 提供同步阻塞接口 `play_audio_blocking(audio_id: int) -> bool`。
3. **通信交互：** 接收到 `audio_id` 后，获取对应文件并通过网络/网络协议传输给 `main_speech.py`，随后阻塞等待，直到接收到 `main_speech.py` 回传的“播放完成”确认信号再解除阻塞返回。

### 3.2 `main_speech.py` 与网页端功能要求
1. **音频接收与推流：** 接收来自程序 A 的音频文件数据，并实时推送到手机端的网页。
2. **前端自动播放：** 手机端网页在接收到完整音频后，自动触发播放（需处理浏览器 Autoplay 权限限制）。
3. **完成状态回调：** 监听网页端 `<audio>` 标签的 `onended` 事件，播放完毕后立即通过 WebSocket 向 `main_speech.py` 发送完成信号，并由 `main_speech.py` 转发给程序 A。

### 3.3 `ws_car_control_V1_3.py` 逻辑集成要求
在以下四个特定场景下，**阻塞式调用**程序 A 播放对应编号的预录制音频。**必须等待播放完成的返回信号，才可执行后续指令（如开始移动、按键检测或语音识别）：**

1. **场景 1（到达学生侧）：** 超声波传感器识别到障碍物并停下后，播放提示音（例：音频 1）。
2. **场景 2（需求响应）：** 学生完成需求表达（通过 ArUco 码或语音识别）后：
   * 若为 **“请求物品”**，播放对应的确认音频（例：音频 2A）。
   * 若为 **“请求老师协助”**，播放对应的确认音频（例：音频 2B）。
3. **场景 3（到达老师侧）：** 小车到达老师位置，等待老师放入物品时，播放提示音频（例：音频 3）。
4. **场景 4（返回学生侧）：** 小车携带物品返回学生面前，等待学生取出物品时，播放提示音频（例：音频 4）。

---

## 4. 异常与边界处理要求

* **网络超时机制：** 在程序 A 的阻塞等待逻辑中加入超时控制（例如最高等待 15 秒），若因网络断开导致未收到 `onended` 回调，应自动超时释放并记录日志，防止小车卡死在当前状态。
* **文件存在性检查：** 程序 A 在发起播放前需校验 `audio_id` 对应的本地音频文件是否存在，若不存在应抛出明确异常。
