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
    RST       - 重置编码器计数

循迹命令（HW-511 模块，完成后均返回 "Done"）:
    TF        - 循迹前进，直到到达交叉路口中央
    TB        - 循迹倒车，直到退至交叉路口中央
    TO        - 循迹前进直到检测到障碍物，返回行驶距离（cm，格式 D:<dist>）
    CL / CR   - 直角弯道沿黑线左转 / 右转 90 度
    PL / PR   - 交叉路口原地左旋 / 右旋 90 度
    PU        - 交叉路口原地旋转 180 度（掉头）

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
    TRACK_TIMEOUT        = 35.0          # 循迹命令最长等待时间（s，Arduino 端 30s 超时 + 余量）

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
    # 循迹命令（HW-511，等待 Done）
    # ------------------------------------------------------------------
    def _track_command(self, cmd: str, desc: str) -> bool:
        """发送循迹命令并阻塞等待 Done / Timeout，返回是否成功"""
        self._write(cmd)
        if self.simulation_mode:
            time.sleep(0.5)
            print(f"  [模拟] {desc} 完成")
            return True
        reply = self._wait_for(lambda l: l in (self.REPLY_DONE, self.REPLY_TIMEOUT),
                               self.TRACK_TIMEOUT, desc=desc)
        return reply == self.REPLY_DONE

    def track_forward(self) -> bool:
        """循迹前进至交叉路口中央（TF 命令），返回 True 表示成功"""
        return self._track_command("TF", "循迹前进")

    def track_backward(self) -> bool:
        """循迹倒车至交叉路口中央（TB 命令），返回 True 表示成功"""
        return self._track_command("TB", "循迹倒车")

    def corner_left(self) -> bool:
        """直角弯道沿黑线左转 90 度（CL 命令）"""
        return self._track_command("CL", "弯道左转")

    def corner_right(self) -> bool:
        """直角弯道沿黑线右转 90 度（CR 命令）"""
        return self._track_command("CR", "弯道右转")

    def pivot_left(self) -> bool:
        """交叉路口原地左旋 90 度（PL 命令）"""
        return self._track_command("PL", "路口左旋")

    def pivot_right(self) -> bool:
        """交叉路口原地右旋 90 度（PR 命令）"""
        return self._track_command("PR", "路口右旋")

    def pivot_u_turn_intersection(self) -> bool:
        """交叉路口原地旋转 180 度掉头（PU 命令）"""
        return self._track_command("PU", "路口掉头180度")

    def pivot_u_turn_line(self) -> bool:
        """直线原地左旋 180 度掉头（PN 命令）"""
        return self._track_command("PN", "直线掉头180度")

    def track_until_obstacle(self) -> float | None:
        """
        循迹前进直到检测到障碍物（TO 命令）。
        返回行驶距离（cm），超时返回 None
        """
        self._write("TO")
        if self.simulation_mode:
            time.sleep(0.5)
            mock_dist = 45.0
            print(f"  [模拟] 循迹检测到障碍物，行驶距离 {mock_dist} cm")
            return mock_dist

        reply = self._wait_for(lambda l: l.startswith(self.REPLY_DISTANCE)
                               or l == self.REPLY_TIMEOUT,
                               self.TRACK_TIMEOUT, desc="循迹障碍物距离")
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
    print("    RST            - 重置编码器")
    print("    --- 循迹命令 (HW-511) ---")
    print("    TF             - 循迹前进至交叉路口")
    print("    TB             - 循迹倒车至交叉路口")
    print("    TO             - 循迹前进直到检测到障碍物")
    print("    CL / CR        - 直角弯道左转 / 右转")
    print("    PL / PR        - 路口原地左旋 / 右旋 90度")
    print("    PU             - 路口原地旋转 180度 掉头")
    print("    PN             - 直线原地旋转 180度 掉头")
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
            # --- 循迹命令（必须先于单字符 T 解析，避免 TF/TB/TO 被当作 T 角度命令）---
            elif upper in ('TF', 'TB', 'CL', 'CR', 'PL', 'PR', 'PU', 'PN'):
                actions = {
                    'TF': car.track_forward,
                    'TB': car.track_backward,
                    'CL': car.corner_left,
                    'CR': car.corner_right,
                    'PL': car.pivot_left,
                    'PR': car.pivot_right,
                    'PU': car.pivot_u_turn_intersection,
                    'PN': car.pivot_u_turn_line,
                }
                ok = actions[upper]()
                print(f"  → {'成功' if ok else '失败/超时'}")
            elif upper == 'TO':
                dist = car.track_until_obstacle()
                if dist is not None:
                    print(f"  → 循迹行驶距离: {dist:.1f} cm")
                else:
                    print("  → 探测超时")
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
