import time
import serial
import serial.tools.list_ports  # pip install pyserial
import sys
import threading
import queue

# 全局线程同步变量
stop_event = threading.Event()  # 紧急停车 S 标志
signal_queue = queue.Queue()     # 存放 O, Q, ok 等业务信号的队列

# ==============================================================================
# 自动查找串口函数
# ==============================================================================
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'ttyUSB' in port.device or 'ttyACM' in port.device:
            return port.device
    return None

# ==============================================================================
# 后台键盘监听线程
# ==============================================================================
def background_input_listener(hw):
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line.upper() == 'S':
                print("\n🛑 [紧急停止] 检测到键盘输入 'S'！正在触发紧急避险...")
                hw.send_arduino_cmd("S") 
                stop_event.set()         
                break
            else:
                signal_queue.put(line)
        except Exception:
            break


# ==============================================================================
# 第一部分：硬件通讯接口层
# ==============================================================================
class CarHardware:
    def __init__(self, baudrate=9600):
        print("==================================================")
        print("[硬件初始化] 正在自动查找 Arduino 串口...")
        self.simulation_mode = False
        self.ser = None
        
        detected_port = find_arduino_port()
        if detected_port is None:
            print("❌ 未找到匹配的 Arduino 硬件端口")
            print("[系统提示] 将自动进入【模拟运行模式】(Simulation Mode)")
            self.simulation_mode = True
        else:
            print(f"✅ 自动匹配到串口: {detected_port}")
            try:
                self.ser = serial.Serial(detected_port, baudrate, timeout=1)
                time.sleep(2) 
                print(f"[硬件初始化] 串口 {detected_port} 连接成功！")
            except Exception as e:
                print(f"❌ 打开串口失败: {e}")
                print("[系统提示] 将自动进入【模拟运行模式】！")
                self.simulation_mode = True
        print("==================================================\n")

    def send_arduino_cmd(self, cmd_str):
        if not self.simulation_mode and self.ser:
            self.ser.write((cmd_str + '\n').encode('utf-8'))
        print(f">>> [下发底盘指令] {cmd_str}")

    def wait_for_done(self):
        if self.simulation_mode:
            for _ in range(10):
                if stop_event.is_set(): raise RuntimeError("用户引发了紧急停车 S")
                time.sleep(0.05)
            print("    [底层反馈] 模拟器: 动作执行完毕 (Done)")
            return
            
        print("    [等待反馈] 正在等待底盘编码器位移到位...")
        while not stop_event.is_set():
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line == "Done":
                    print("    [底层反馈] Arduino: 动作执行完毕 (Done)")
                    break
            time.sleep(0.02)
        if stop_event.is_set():
            raise RuntimeError("在等待底盘响应时检测到紧急停车指令 S")

    def start_obstacle_detection(self):
        self.send_arduino_cmd("O")
        if self.simulation_mode:
            for _ in range(20):
                if stop_event.is_set(): raise RuntimeError("用户引发了紧急停车 S")
                time.sleep(0.05)
            mock_dist = 45.0
            print(f"    [底层反馈] 模拟器: 遇到障碍物，本次探测前进了 {mock_dist} cm")
            return mock_dist
            
        print("    [探测中] 小车持续前进，等待底盘返回 'D:距离' 信号...")
        while not stop_event.is_set():
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("D:"):
                    dist_str = line.split(":")[1]
                    print(f"    [底层反馈] Arduino: 遇到障碍物，本次探测前进了 {dist_str} cm")
                    return float(dist_str)
            time.sleep(0.02)
        if stop_event.is_set():
            raise RuntimeError("在超声波探测时检测到紧急停车指令 S")


# ==============================================================================
# 第二部分：运动控制层 (基于降维后的绝对分叉点拓扑逻辑)
# ==============================================================================
class CarController:
    def __init__(self, hardware):
        self.hw = hardware
        
        # 核心拓扑位置：0, 1, 2, 3 分别代表分叉点 l1, l2, l3, l4
        self.current_idx = 0       
        
        # 路径参数 (单位：cm, deg)
        self.LENGTH0_0 = 20.0   
        self.LENGTH0_1 = 20.0   
        self.LOOPS = [20.0, 25.0, 30.0, 25.0]  # l1, l2, l3, l4 的单段距离
        
        self.TURN_90_ANGLE = 90.0    

    def execute_move(self, action_type, amount):
        if stop_event.is_set(): raise RuntimeError("检测到紧急停车指令 S")
        if amount <= 0: return
        
        if action_type == 'F':    self.hw.send_arduino_cmd(f"G {amount}")
        elif action_type == 'B':  self.hw.send_arduino_cmd(f"G -{amount}")
        elif action_type == 'L':  self.hw.send_arduino_cmd(f"T {amount}")
        elif action_type == 'R':  self.hw.send_arduino_cmd(f"T -{amount}")
            
        self.hw.wait_for_done()
        time.sleep(0.1) 

    def get_accumulated_dist(self):
        """计算当前分叉点距离主回路入口 E 的绝对物理距离"""
        return sum(self.LOOPS[:self.current_idx + 1])

    def reverse_path_to_start_fork(self, turn_direction):
        """
        核心简化：因为车头永远朝前，无论何时回到起点，计算方法完全一致！
        """
        print("\n==================================================")
        print(f"🚀 [分叉点回溯] 当前位于分叉点 l{self.current_idx+1}，正在退回起点...")
        
        # 1. 回正车头：从面向侧边栏转回平行于主干道向前（朝北）的方向
        if turn_direction == 'L':   self.execute_move('R', self.TURN_90_ANGLE)
        elif turn_direction == 'R': self.execute_move('L', self.TURN_90_ANGLE)
            
        # 2. 绝对位移倒车：因为车头朝前，直接向后倒车累计距离，就能精准降落到入口 E 
        accum_dist = self.get_accumulated_dist()
        self.execute_move('B', accum_dist)
            
        # 3. 从入口 E 倒车退回原始 startpoint
        self.execute_move('L', self.TURN_90_ANGLE)
        self.execute_move('B', self.LENGTH0_1)
        self.execute_move('R', self.TURN_90_ANGLE)
        self.execute_move('B', self.LENGTH0_0)
        
        print("✅ [回到起点] 小车成功倒车退回最初的 startpoint！")
        print("==================================================\n")

    def replay_path_to_fork(self, turn_direction):
        """从绝对起点 startpoint 重新驶回当前触发中断的分叉点"""
        print("\n==================================================")
        print(f"🚀 [路径复现] 正在重新前往分叉点 l{self.current_idx+1} 的探测现场...")
        
        # 1. 正向通过引导路段前往入口 E
        self.execute_move('F', self.LENGTH0_0)
        self.execute_move('L', self.TURN_90_ANGLE)
        self.execute_move('F', self.LENGTH0_1)
        self.execute_move('R', self.TURN_90_ANGLE)
        
        # 2. 直行走到对应的分叉点路口
        accum_dist = self.get_accumulated_dist()
        self.execute_move('F', accum_dist)
            
        # 3. 重新把车头偏转对准侧边栏
        if turn_direction == 'L':   self.execute_move('L', self.TURN_90_ANGLE)
        elif turn_direction == 'R': self.execute_move('R', self.TURN_90_ANGLE)
            
        print("✅ [抵达现场] 小车已重新回到分叉点路口！")
        print("==================================================\n")

    def wait_for_signal(self, valid_signals, timeout=None):
        print(f"\n[等待指令] 正在监听控制信号 {valid_signals} (输入 S 可随时紧急停车) ...")
        
        if timeout is not None:
            print(f"    [超时设置] {timeout} 秒内未收到有效信号将自动继续。")
            start_time = time.time()

        #if not hasattr(self, "_btn_state"):
        #    self._btn_state = "ready1"

        while True:
            if stop_event.is_set(): 
                raise RuntimeError("在等待业务信号时检测到紧急停车指令 S")
            
            ###
            #if not self.hw.simulation_mode and self.hw.ser and self.hw.ser.in_waiting > 0:
            #    try:
            #        line = self.hw.ser.readline().decode('utf-8').strip()
            #        if line == "BTN":
            #            print(f"\n🔘 [硬件按键] 收到 Arduino 触发！当前映射为: '{self._btn_state}'")
            #            # 自动将映射好的信号塞入队列
            #            signal_queue.put(self._btn_state)
            #            
            #            # 状态反转：为下一次按压做准备
            #            if self._btn_state == "ready1":
            #                self._btn_state = "ready2"
            #            else:
            #                self._btn_state = "ready1"
            #    except Exception as e:
            #        pass
            ###
            
            try:
                sig = signal_queue.get(timeout=0.1)
                if sig in valid_signals:
                    print(f"[捕获成功] 系统收到业务指令: '{sig}'，继续向下执行。")
                    return sig
                else:
                    print(f"[输入错误] 收到的指令 '{sig}' 不在允许列表 {valid_signals} 中，请重新输入！")
            except queue.Empty:
                if timeout is not None and (time.time() - start_time) >= timeout:
                    print(f"[超时] 在 {timeout} 秒内未收到有效信号，自动继续。")
                    return None
                else:
                    continue

    def obstacle_detection_routine(self, turn_direction):
        """遭遇 O 信号后的标准化往返流（由于坐标不反转，此段逻辑极其稳定）"""
        print(" -> [流程触发] 启动超声波探路...")
        detect_distance = self.hw.start_obstacle_detection()
        print("arrived_student")
        
        self.wait_for_signal(["go_teacher"])
        
        # 1. 退出侧边栏，倒车回到分叉点中心
        print(" -> [退回分叉点] 正在倒车脱离侧边栏...")
        self.execute_move('B', detect_distance)
        
        # 2. 倒回真正的起点
        self.reverse_path_to_start_fork(turn_direction)
        print("arrived_teacher")
        
        self.wait_for_signal(["return_student"])
        
        # 3. 重新开回分叉点
        self.replay_path_to_fork(turn_direction)
        
        # 4. 再次深入侧边栏去靠近障碍物
        print(" -> [前行至现场] 重新驶入侧边栏靠近障碍物点...")
        self.execute_move('F', detect_distance)
        print("arrived_student")
        
        self.wait_for_signal(["return_patrol"])
        
        # 5. 最终姿态补偿：退回分叉点中心，并将车头转回直行状态
        print(" -> [姿态补偿] 倒退回该路段最初的旋转拐角处...")
        self.execute_move('B', detect_distance)
        
        print(" -> [姿态补偿] 执行车头方向回正...")
        if turn_direction == 'L':   self.execute_move('R', self.TURN_90_ANGLE)
        elif turn_direction == 'R': self.execute_move('L', self.TURN_90_ANGLE)
        print("route_rejoined")

    def perform_double_side_detection(self):
        """在当前分叉点执行标准的左右双侧探测"""
        # ====== 左侧检测 ======
        print(f" -> 执行左转探测")
        self.execute_move('L', self.TURN_90_ANGLE)
        
        sig = self.wait_for_signal(["approach_student"], timeout=5)
        if sig == "approach_student":
            self.obstacle_detection_routine(turn_direction='L')
            print(" -> [左侧] 完结。准备进入右侧同步探测...")
            self.execute_move('R', self.TURN_90_ANGLE)
            sig_r = self.wait_for_signal(["approach_student"], timeout=5)
            if sig_r == "approach_student":   self.obstacle_detection_routine(turn_direction='R')
            else: self.execute_move('L', self.TURN_90_ANGLE)
                
        else:
            print(" -> [左侧] 无障碍物，向右回正。")
            self.execute_move('R', self.TURN_90_ANGLE)
            
            print(" -> 准备进入右侧同步探测...")
            self.execute_move('R', self.TURN_90_ANGLE)
            sig_r = self.wait_for_signal(["approach_student"], timeout=5)
            if sig_r == "approach_student":   self.obstacle_detection_routine(turn_direction='R')
            else:
                print(" -> [右侧] 无障碍物，向左回正。")
                self.execute_move('L', self.TURN_90_ANGLE)


# ==============================================================================
# 第三部分：主循环业务流 (正向推进 + 逆向倒车 往复流)
# ==============================================================================
    def run(self):
        print(">>>>>>>>>> 算法启动：开始执行引导路线 <<<<<<<<<<\n")
        self.execute_move('F', self.LENGTH0_0)
        self.execute_move('L', self.TURN_90_ANGLE)
        self.execute_move('F', self.LENGTH0_1)
        self.execute_move('R', self.TURN_90_ANGLE)
        
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n==================== 第 {loop_count} 轮 正反往复扫描循环 ====================")
            
            # --------- 阶段 1：正向推进扫描 (l1 -> l4) ---------
            print("\n➡️ >>>>>> [正向推进阶段] >>>>>>")
            for idx in range(4):
                self.current_idx = idx  
                l_dist = self.LOOPS[idx]
                print(f"\n--- [正向] 前进至路段 l{idx+1} 中心，行驶里程: {l_dist} cm ---")
                
                self.execute_move('F', l_dist)
                self.perform_double_side_detection()
                
            # --------- 阶段 2：逆向倒车扫描 (l4 -> l1) ---------
            print("\n⬅️ <<<<<< [逆向倒车阶段] <<<<<<")
            # 从 l4 分叉点开始往回倒车，依次在 l3, l2, l1 分叉点驻留并探测
            # 当 idx=2 时，代表小车从 l4 倒车回退到 l3 分叉点，依此类推
            for idx in [2, 1, 0]:
                l_dist = self.LOOPS[idx + 1] # 计算相邻分叉点之间的倒车距离
                print(f"\n--- [逆向] 倒车回退至路段 l{idx+1} 中心，倒退里程: {l_dist} cm ---")
                
                self.execute_move('B', l_dist)
                self.current_idx = idx  # 精准更新当前拓扑索引
                self.perform_double_side_detection()
            
            # --------- 阶段 3：倒车回主回路入口 E ---------
            print(f"\n--- [逆向完结] 正在从小车所在的 l1 彻底倒回入口 E: {self.LOOPS[0]} cm ---")
            self.execute_move('B', self.LOOPS[0])
            print("\n🏁 [本轮结束] 小车已无转向倒回入口 E 点，即将开启下一轮往复！")


if __name__ == "__main__":
    while True:
        start_cmd = input("👉 请输入 'start' 开始整个完整的过程: ").strip()
        if start_cmd.lower() == 'start':
            print("▶️ [启动成功] 系统控制权限已激活，流程正式开始！")
            print("💡 [提示] 在终端随时敲击 'S' 并回车可命令小车紧急刹车并退出程序。\n")
            break
        else:
            print("[错误] 指令未识别，必须先输入 'start'。")

    hw = CarHardware(baudrate=9600)
    car = CarController(hw)
    
    listener_thread = threading.Thread(target=background_input_listener, args=(hw,), daemon=True)
    listener_thread.start()
    
    try:
        car.run()
    except RuntimeError as e:
        print(f"\n🛑 [业务流中断] {e}")
        print("[系统退出] 小车已安全制动，程序退出。")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[系统提示] 强制退出。")
        sys.exit(0)