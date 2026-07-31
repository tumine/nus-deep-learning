import asyncio
import websockets
import json
import serial
import serial.tools.list_ports
import time
import sys

# ================== 自动查找 Arduino 串口 ==================
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'ttyUSB' in port.device or 'ttyACM' in port.device:
            return port.device
    return None

SERIAL_PORT = find_arduino_port()
if SERIAL_PORT is None:
    print("❌ 未找到 Arduino，请检查 USB 连接")
    print("   或手动指定 SERIAL_PORT = '/dev/ttyUSB0'")
    exit(1)

print(f"✅ 使用串口: {SERIAL_PORT}")

# ================== 打开串口 ==================
try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.1)
    time.sleep(2)  # 等待 Arduino 复位
    print("✅ 串口已打开")
except Exception as e:
    print(f"❌ 打开串口失败: {e}")
    exit(1)

# ================== 指令映射 ==================
# 映射表：全称 → 单字符指令（用于 WebSocket 和本地输入）
CMD_MAP = {
    "forward": 'F',
    "backward": 'B',
    "left": 'L',
    "right": 'R',
    "stop": 'S',
    "auto": 'A'    # 切换模式
}

# 有效的单字符指令（直接发送）
VALID_CHARS = {'F', 'B', 'L', 'R', 'S', 'A'}

def send_to_arduino(command):
    """
    将指令（单字符或全称）转换为 Arduino 可识别的字符并发送。
    支持：单字符（F/B/L/R/S/A）或全称（forward/backward/left/right/stop/auto）
    """
    if not command:
        return

    # 1. 如果是单字符且合法，直接使用
    ch = None
    if len(command) == 1:
        ch = command.upper()
        if ch not in VALID_CHARS:
            ch = None

    # 2. 如果未匹配，尝试从映射表中查找
    if ch is None:
        mapped = CMD_MAP.get(command.lower())
        if mapped:
            ch = mapped

    if ch is None:
        print(f"⚠️ 未知指令: {command}")
        return

    print(f">> 发送指令: {command} -> '{ch}'")
    try:
        ser.write((ch + '\n').encode())
        ser.reset_input_buffer()  # 清空缓冲区防止残留
    except Exception as e:
        print(f"❌ 串口发送失败: {e}")

# ================== WebSocket 服务端 ==================
async def car_controller(websocket, path):
    print(f"📱 手机已连接！IP: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get("action")
                if command:
                    send_to_arduino(command)
            except json.JSONDecodeError:
                print(f"⚠️ 收到非 JSON 消息: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("📱 手机已断开连接")
    finally:
        # 连接断开时自动停止小车
        send_to_arduino("stop")

# ================== 本地命令行输入 ==================
async def stdin_reader():
    """在终端中读取用户输入（非阻塞），每行一条指令"""
    loop = asyncio.get_running_loop()
    print("\n💻 本地控制模式：输入 F/B/L/R/S/A 或 forward/backward/left/right/stop/auto 后回车")
    while True:
        try:
            # 使用线程池执行阻塞的 input()
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF (Ctrl+D)
                break
            cmd = line.strip()
            if cmd:
                send_to_arduino(cmd)
        except Exception as e:
            print(f"❌ 本地输入读取错误: {e}")
            break

# ================== 启动服务 ==================
async def main():
    print("🚀 WebSocket 服务已启动")
    print("📡 监听端口: 8765")
    print("📱 请在手机浏览器中打开 control.html")
    print("💻 本地控制: 在终端输入 F/B/L/R/S/A (或全称) 后回车")
    print("⏹️  按 Ctrl+C 停止服务\n")

    # 启动 WebSocket 服务
    async with websockets.serve(car_controller, "0.0.0.0", 8765):
        # 启动本地输入监听任务
        stdin_task = asyncio.create_task(stdin_reader())
        try:
            # 无限等待，直到被中断
            await asyncio.Future()
        finally:
            # 取消本地输入任务，避免残留
            stdin_task.cancel()
            try:
                await stdin_task
            except asyncio.CancelledError:
                pass

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n🛑 服务已停止")
finally:
    ser.close()
    print("🔌 串口已关闭")