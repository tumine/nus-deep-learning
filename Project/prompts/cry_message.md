## Date: 23 July, 2026

---

### 需求
在 cry_detect 目录下的 voice_transmission_server_panns.py 中，把检测到的哭声事件以消息的方式通过 TCP 连接传输给教师端。

### 交付要求
修改后的 Python 程序，实现把检测到的哭声事件以消息的方式通过 TCP 连接传输给教师端。

### 部署位置
代码部署在笔记本电脑上。

### 消息格式
一个 JSON 字符串，具体包含的字段：
- `message_id`，每条消息的唯一 ID 编号
- `axis_x`，事件发生位置的 x 坐标
- `axis_y`，事件发生位置的 y 坐标
- `request`，请求类型（物品，教师协助）
- `description`，请求详细信息（具体物品名称，请求教师协助的原因）

### 参考程序
- `cry_detect/voice_transmission_server_panns.py` 是部署哭声检测模型的程序，当检测到哭声时会发出哭声检测告警。
- `teacher_client.py` 是教师客户端，包括**后端消息处理**和**前端网页展示逻辑**。
- `teacher_client_test.py` 是测试教师客户端运行功能的程序，可以按 [消息格式] 把消息发送到教师端。

### 程序实现详细要求
1. 在检测到哭声时（控制台发出哭声检测告警），新增消息发送逻辑，按 [消息格式] 打包一条消息发送到教师端。
2. `axis_x, axis_y` 参数悬空，给定任意无效值即可。
