#!/usr/bin/env python3
"""
测试脚本 - 向教师端监控程序发送模拟小车数据
用法: python test_client.py
依赖: pip install websockets
"""

import asyncio
import json
import random
import websockets

# 服务器 WebSocket 地址（默认与 teacher_client.py 同机）
WS_URL = "ws://127.0.0.1:8000/ws"

# 模拟数据生成配置
NUM_MESSAGES = 10          # 发送消息数量
SEND_INTERVAL = 1.0        # 每条消息间隔（秒）

# 物品名称列表（随机选择）
ITEM_NAMES = ["扳手", "螺丝刀", "锤子", "电钻", "胶带", "电池", "灯泡", "电缆", "电阻", "电容", "芯片", "传感器"]
# 协助原因列表
HELP_REASONS = [
    "程序运行卡顿，需要指导",
    "机械臂夹持不稳定",
    "传感器读数异常",
    "电机过热报警",
    "通讯中断，需要重启",
    "路径规划失败，请求人工干预",
    "电源电压偏低",
    "视觉识别错误"
]

def generate_random_message(index: int) -> dict:
    """生成一条模拟消息"""
    request_type = random.choice(["物品", "教师协助"])
    if request_type == "物品":
        desc = random.choice(ITEM_NAMES)
    else:
        desc = random.choice(HELP_REASONS)

    return {
        "message_id": f"MSG-{index:04d}",
        "axis_x": round(random.uniform(0, 100), 2),
        "axis_y": round(random.uniform(0, 100), 2),
        "request": request_type,
        "description": desc
    }

async def send_messages():
    """连接 WebSocket 并发送消息"""
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"✅ 已连接到 {WS_URL}")
            print(f"📤 准备发送 {NUM_MESSAGES} 条模拟消息...\n")

            for i in range(1, NUM_MESSAGES + 1):
                msg = generate_random_message(i)
                # 转为 JSON 字符串
                json_str = json.dumps(msg, ensure_ascii=False)
                await websocket.send(json_str)

                # 打印发送的信息（含当前时间）
                print(f"[{i:>2}] 发送: {json_str}")
                # 等待应答（如果需要，也可以不等待，因为服务端不回显）
                # 这里只是延迟一下，以便观察页面更新
                await asyncio.sleep(SEND_INTERVAL)

            print("\n✅ 所有消息发送完成。")
    except websockets.exceptions.ConnectionClosedError:
        print("❌ 连接意外关闭，请确保 teacher_client.py 正在运行。")
    except ConnectionRefusedError:
        print("❌ 连接被拒绝，请确保 teacher_client.py 已启动并监听 8000 端口。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(send_messages())