"""
serial_car_control.py
=====================
树莓派通过串口控制 Arduino 小车模块。

支持的串口命令（与 newrun.ino 对应）:
    F         - 持续前进
    B         - 持续后退
    L         - 持续左转
    R         - 持续右转
    S         - 紧急停车
    O         - 前进直到检测到障碍物，返回前近距离（cm）
    G <dist>  - 移动指定距离（cm），正=前进，负=后退，完成后返回 "Done"
    T <angle> - 旋转指定角度（度），正=左转，负=右转，完成后返回 "Done"
    M         - 陀螺仪诊断（不开电机，手动转动小车，约 10s，结束时返回 "Debug done"）
    RST       - 重置编码器计数

用法:
    python serial_car_control.py          # 交互模式
    或:
    from serial_car_control import CarSerial
    car = CarSerial()
    car.go_forward()
    car.move_distance(50)
    car.stop()
"""

import time
import serial
import serial.tools.list_ports
import sys


# ==============================================================================
# 辅助函数
# ==============================================================================
def find_arduino_port():
    """自动查找 Arduino 串口"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if ('Arduino' in port.description
                or 'ttyUSB' in port.device
                or 'ttyACM' in port.device):
            return port.device
    return None


# ==============================================================================
# CarSerial — 串口控制类
# ==============================================================================
class CarSerial:
    """通过串口控制 Arduino 小车，自动检测端口 + 模拟模式降级"""

    # ---------- 常量 ----------
    # 从 Arduino 固件读取的返回值约定
    REPLY_DONE           = "Done"
    REPLY_DISTANCE       = "D:"          # 前缀，后面跟距离值
    REPLY_ENCODERS_RESET = "Encoders reset"
    REPLY_GYRO_DEBUG     = "Debug done"
    REPLY_STOPPED        = "Stopped"
    REPLY_FORWARD        = "Forward"
    REPLY_BACKWARD       = "Backward"
    REPLY_TURN_LEFT      = "Turn Left"
    REPLY_TURN_RIGHT     = "Turn Right"
    REPLY_INVALID_CMD    = "Unknown command"
    REPLY_TIMEOUT        = "Timeout"

    DEFAULT_BAUDRATE     = 9600
    DEFAULT_TIMEOUT      = 1.0
    MOVE_TIMEOUT         = 35.0          # G / T 命令最长等待时间（s）
    OBSTACLE_TIMEOUT     = 60.0          # O 命令最长等待时间（s）
    GYRO_DEBUG_TIMEOUT   = 15.0          # M 命令最长等待时间（s，固件约 10s）

    def __init__(self, baudrate=None, port=None, simulation_mode=False):
        """
        参数:
            baudrate: 串口波特率，默认 9600
            port: 手动指定串口（如 '/dev/ttyACM0'）。为 None 则自动查找
            simulation_mode: True 则强制模拟模式（不连接硬件）
        """
        baudrate = baudrate or self.DEFAULT_BAUDRATE
        self.simulation_mode = simulation_mode
        self.ser = None

        if simulation_mode:
            print("[CarSerial] 模拟模式已开启，不会连接硬件")
            return

        # --- 自动查找 / 手动指定 ---
        detected = port or find_arduino_port()
        if detected is None:
            print("[CarSerial] 未找到 Arduino 端口 → 自动进入模拟模式")
            self.simulation_mode = True
            return

        try:
            self.ser = serial.Serial(detected, baudrate, timeout=self.DEFAULT_TIMEOUT)
            time.sleep(2)  # Arduino 复位后需要稳定
            print(f"[CarSerial] 串口 {detected} 连接成功 (baudrate={baudrate})")
        except Exception as e:
            print(f"[CarSerial] 打开串口失败: {e} → 自动进入模拟模式")
            self.simulation_mode = True

    # ------------------------------------------------------------------
    # 底层通讯
    # ------------------------------------------------------------------
    def _write(self, cmd: str):
        """向 Arduino 发送一行命令"""
        if not self.simulation_mode and self.ser:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + '\n').encode('utf-8'))
        print(f"  [串口发送] {cmd}")

    def _read_line(self, timeout=None) -> str | None:
        """读取一行串口回复（阻塞，直到读到行或超时）"""
        if self.simulation_mode:
            time.sleep(0.1)
            return None
        if not self.ser:
            return None

        start = time.time()
        while timeout is None or (time.time() - start) < timeout:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    print(f"  [串口收到] {line}")
                    return line
            time.sleep(0.01)
        return None

    def _wait_for(self, predicate, timeout: float, desc: str = "") -> str | None:
        """轮询等待直到 predicate(line) 为 True 或超时"""
        start = time.time()
        while (time.time() - start) < timeout:
            line = self._read_line(timeout=0.5)
            if line and predicate(line):
                return line
        print(f"  [警告] 等待 {desc} 超时 ({timeout}s)")
        return None

    # ------------------------------------------------------------------
    # 基础运动命令（持续模式，不等待 Done）
    # ------------------------------------------------------------------
    def go_forward(self):
        """持续前进（F）"""
        self._write("F")

    def go_backward(self):
        """持续后退（B）"""
        self._write("B")

    def turn_left(self):
        """持续左转（L）"""
        self._write("L")

    def turn_right(self):
        """持续右转（R）"""
        self._write("R")

    def stop(self):
        """紧急停车（S）"""
        self._write("S")

    # ------------------------------------------------------------------
    # 精确运动命令（等待 Done）
    # ------------------------------------------------------------------
    def move_distance(self, dist_cm: float) -> bool:
        """
        移动指定距离（G 命令）。
        dist_cm > 0 → 前进，< 0 → 后退
        返回 True 表示成功，False 表示超时 / 失败
        """
        if dist_cm == 0:
            return True
        self._write(f"G {dist_cm}")
        if self.simulation_mode:
            time.sleep(0.5)
            print(f"  [模拟] 移动 {dist_cm} cm 完成")
            return True
        reply = self._wait_for(lambda l: l in (self.REPLY_DONE, self.REPLY_TIMEOUT),
                               self.MOVE_TIMEOUT, desc="G 命令完成")
        return reply == self.REPLY_DONE

    def turn_angle(self, angle_deg: float):
        """
        旋转指定角度（T 命令）。
        angle_deg > 0 → 左转，< 0 → 右转
        返回 True 表示成功
        """
        if abs(angle_deg) < 0.5:
            return True
        self._write(f"T {angle_deg}")
        if self.simulation_mode:
            time.sleep(0.5)
            print(f"  [模拟] 旋转 {angle_deg}° 完成")
            return True
        reply = self._wait_for(lambda l: l in (self.REPLY_DONE, self.REPLY_TIMEOUT),
                               self.MOVE_TIMEOUT, desc="T 命令完成")
        return reply == self.REPLY_DONE

    # ------------------------------------------------------------------
    # 超声波探测
    # ------------------------------------------------------------------
    def drive_until_obstacle(self) -> float | None:
        """
        前进直到检测到障碍物（O 命令）。
        返回前近距离（cm），超时返回 None
        """
        self._write("O")
        if self.simulation_mode:
            time.sleep(0.5)
            mock_dist = 45.0
            print(f"  [模拟] 检测到障碍物，距离 {mock_dist} cm")
            return mock_dist

        reply = self._wait_for(lambda l: l.startswith(self.REPLY_DISTANCE),
                               self.OBSTACLE_TIMEOUT, desc="障碍物距离")
        if reply and reply.startswith(self.REPLY_DISTANCE):
            try:
                return float(reply.split(":")[1])
            except (IndexError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------
    # 编码器重置
    # ------------------------------------------------------------------
    def reset_encoders(self):
        """重置编码器计数（RST 命令）"""
        self._write("RST")

    # ------------------------------------------------------------------
    # 陀螺仪诊断
    # ------------------------------------------------------------------
    def debug_gyro(self, collect: bool = False) -> list | None:
        """
        陀螺仪诊断（M 命令）。
        不开电机，手动转动小车约 10s，固件周期打印
        "GyroZ:/YawInt:/Err:"，结束时返回 "Debug done"。

        参数:
            collect: 为 True 时收集并打印每一行诊断数据；为 False 仅等待结束。
        返回:
            collect=True  → 诊断行列表（list[str]），超时返回 None
            collect=False → 是否成功完成 (bool)
        """
        self._write("M")
        if self.simulation_mode:
            time.sleep(0.5)
            print("  [模拟] 陀螺仪诊断完成")
            if collect:
                return ["GyroZ:0.0 YawInt:0.0 Err:0"]
            return True

        samples = []
        start = time.time()
        while (time.time() - start) < self.GYRO_DEBUG_TIMEOUT:
            line = self._read_line(timeout=0.5)
            if not line:
                continue
            if line.startswith("GyroZ:") or line == self.REPLY_GYRO_DEBUG:
                if collect:
                    samples.append(line)
                if line == self.REPLY_GYRO_DEBUG:
                    return samples if collect else True
        print(f"  [警告] 等待 {self.REPLY_GYRO_DEBUG} 超时 ({self.GYRO_DEBUG_TIMEOUT}s)")
        return None if collect else False

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[CarSerial] 串口已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ==============================================================================
# 交互式命令行
# ==============================================================================
def interactive():
    """简单的交互式命令终端，方便手动调试"""
    print("=" * 50)
    print("  小车串口控制 - 交互模式")
    print("=" * 50)
    print("  命令:")
    print("    F / B / L / R  - 持续运动")
    print("    S              - 紧急停车")
    print("    G <距离cm>     - 前进指定距离（负数为后退）")
    print("    T <角度deg>    - 旋转指定角度（负数为右转）")
    print("    O              - 前进直到检测到障碍物")
    print("    M              - 陀螺仪诊断（约 10s，手动转动小车）")
    print("    RST            - 重置编码器")
    print("    quit / q       - 退出")
    print("=" * 50)

    car = CarSerial()

    try:
        while True:
            try:
                raw = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue

            upper = raw.upper()
            if upper in ('QUIT', 'Q', 'EXIT'):
                break
            elif upper == 'F':
                car.go_forward()
            elif upper == 'B':
                car.go_backward()
            elif upper == 'L':
                car.turn_left()
            elif upper == 'R':
                car.turn_right()
            elif upper == 'S':
                car.stop()
            elif upper == 'O':
                dist = car.drive_until_obstacle()
                if dist is not None:
                    print(f"  → 障碍物距离: {dist:.1f} cm")
                else:
                    print("  → 探测超时")
            elif upper.startswith('G'):
                try:
                    dist = float(raw[1:].strip())
                    ok = car.move_distance(dist)
                    print(f"  → {'成功' if ok else '失败/超时'}")
                except ValueError:
                    print("  → 用法: G 30   或   G -20")
            elif upper.startswith('T'):
                try:
                    angle = float(raw[1:].strip())
                    ok = car.turn_angle(angle)
                    print(f"  → {'成功' if ok else '失败/超时'}")
                except ValueError:
                    print("  → 用法: T 90   或   T -45")
            elif upper == 'RST':
                car.reset_encoders()
            elif upper == 'M':
                samples = car.debug_gyro(collect=True)
                if samples is None:
                    print("  → 诊断超时")
                else:
                    print(f"  → 诊断完成，共 {len(samples)} 条数据；末行: {samples[-1]}")
            else:
                print(f"  → 未知命令: {raw}")
    finally:
        car.close()
        print("\n已退出。")


# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    interactive()
