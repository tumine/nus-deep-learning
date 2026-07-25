import time
import serial
import serial.tools.list_ports  
import sys
import threading
import queue

import socket
global_conn = None

# 全局线程同步变量
stop_event = threading.Event()  # 紧急停车 S 标志
signal_queue = queue.Queue()     # 存放 O, Q, ok 等业务信号的队列

# 【新增】网络连接就绪标志
network_ready_event = threading.Event()

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
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue
            if line.upper() == 'S':
                print("\n🛑 [小车终端暂停] 检测到键盘输入 'S'！正在控制小车底层刹车暂停...")
                hw.send_arduino_cmd("S") 
                stop_event.set()         
                # break
            else:
                signal_queue.put(line)
        except Exception:
            break

# ==============================================================================
# 后台线程 2：TCP 网络监听线程（新增：用于接收电脑发来的控制指令）
# ==============================================================================
def tcp_server_listener(hw):
    global global_conn
    HOST = '0.0.0.0'  # 监听所有可用 IP
    PORT = 9999       # 自定义通信端口

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"\n📡 [网络通信] TCP 服务端已启动，正在端口 {PORT} 等待电脑端连接...")

    while True:
        try:
            server_socket.settimeout(1.0)
            conn, addr = server_socket.accept()
            global_conn = conn
            print(f"\n✅ [网络通信] 电脑端已连接: {addr}")

            # 【新增】通知主线程：网络已就绪
            network_ready_event.set()
            
            buffer = ""
            while True:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    print("⚠️ [网络通信] 电脑端已断开连接")
                    break
                
                buffer += data
                # 使用 \n 分割，处理 TCP 粘包问题
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    print(f"📥 [收到网络指令]: '{line}'")
                    
                    if line.upper() == 'S':
                        print("\n🛑 [网络控制暂停] 收到电脑端发来的 'S' 指令！正在控制小车底层刹车暂停...")
                        hw.send_arduino_cmd("S") 
                        stop_event.set()
                        # break
                    else:
                        # 业务指令放入队列，供小车主逻辑读取
                        signal_queue.put(line)
                        
        except socket.timeout:
            continue
        except Exception as e:
            print(f"❌ 网络连接异常: {e}")
            global_conn = None

def send_status_to_pc(status_msg):
    """用于向电脑端发送小车当前状态"""
    global global_conn
    print(f"📤 [发送状态至电脑]: {status_msg}")
    if global_conn:
        try:
            # 加上换行符防止粘包
            global_conn.sendall((status_msg + '\n').encode('utf-8'))
        except Exception as e:
            print(f"❌ 发送状态失败: {e}")


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
                elif line == "Obstacle Stop":
                    print("\n🛑 [底层安全警报] 硬件触发全局障碍物急停！正在同步中断 Python 业务流...")
                    stop_event.set() # 激活全局打断标志
                    raise RuntimeError("底层硬件检测到障碍物，自动触发急停")
            time.sleep(0.02)
        if stop_event.is_set():
            raise RuntimeError("在等待底盘响应时检测到紧急停车指令 S")

    def start_obstacle_detection(self):
        # 使用 Arduino 中新加的循迹探障指令 TO，确保分支内仍能沿线行驶
        self.send_arduino_cmd("TO")
        if self.simulation_mode:
            for _ in range(20):
                if stop_event.is_set(): raise RuntimeError("用户引发了紧急停车 S")
                time.sleep(0.05)
            mock_dist = 45.0
            print(f"    [底层反馈] 模拟器: 遇到障碍物，本次探测前进了 {mock_dist} cm")
            return mock_dist
            
        print("    [探测中] 小车正沿线寻迹前进，等待底盘返回 'D:距离' 信号...")
        while not stop_event.is_set():
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("D:"):
                    dist_str = line.split(":")[1]
                    print(f"    [底层反馈] Arduino: 遇到障碍物，本次探测前进了 {dist_str} cm")
                    # 消耗掉 Arduino 随后发送的 "Done"
                    time.sleep(0.1)
                    while self.ser.in_waiting > 0:
                        self.ser.readline()
                    return float(dist_str)
                # 【新增】捕获底层发来的障碍物急停信号
                elif line == "Obstacle Stop":
                    print("\n🛑 [底层安全警报] 硬件触发全局障碍物急停！正在同步中断 Python 业务流...")
                    stop_event.set()
                    raise RuntimeError("底层硬件检测到障碍物，自动触发急停")
            time.sleep(0.02)
        if stop_event.is_set():
            raise RuntimeError("在超声波探测时检测到紧急停车指令 S")


# ==============================================================================
# 第二部分：绝对坐标与航向状态机控制层 (彻底解决转向迷失问题)
# ==============================================================================
# 定义绝对方向常量
NORTH = 0  # 主干道前方
EAST = 1   # 主干道右侧 (包含起点的入口通道)
SOUTH = 2  # 主干道后方
WEST = 3   # 主干道左侧

class CarController:
    def __init__(self, hardware):
        self.hw = hardware
        self.pos_x = -1  # 初始位置：起点 (老师位置)
        self.pos_y = 0   
        self.heading = EAST # 初始朝向：车头面朝东 (指向主干道入口)
        
    def execute_cmd(self, cmd_str):
        if stop_event.is_set(): raise RuntimeError("检测到紧急停车指令 S")
        self.hw.send_arduino_cmd(cmd_str)
        self.hw.wait_for_done()
        time.sleep(0.1)

    def go_detect_obstacle(self):
        """执行循迹探测并获取距离"""
        if stop_event.is_set(): raise RuntimeError("检测到紧急停车指令 S")
        dist = self.hw.start_obstacle_detection()
        time.sleep(0.1)
        return dist

    def turn_to(self, target_heading):
        """智能计算并转向到指定的绝对朝向"""
        if self.heading == target_heading:
            return
            
        diff = (target_heading - self.heading) % 4
        if diff == 1:
            self.execute_cmd('PR') # 右转 90度
        elif diff == 2:
            self.execute_cmd('PU') # 十字路口掉头 180度
        elif diff == 3:
            self.execute_cmd('PL') # 左转 90度
            
        self.heading = target_heading

    def move_to_y(self, target_y):
        """沿主干道行驶到目标 Y 坐标 (交叉口序号)"""
        if self.pos_x != 0:
            raise RuntimeError("当前不在主干道中心，无法进行纵向移动！")
            
        if target_y > self.pos_y:
            self.turn_to(NORTH)
            for _ in range(target_y - self.pos_y):
                self.execute_cmd('TF')
            self.pos_y = target_y
            
        elif target_y < self.pos_y:
            self.turn_to(SOUTH)
            for _ in range(self.pos_y - target_y):
                self.execute_cmd('TF')
            self.pos_y = target_y

    def continuous_move_to_y(self, target_y):
        """
        【新增】连续性移动指令：利用 TF1, TF2, TF3, TF4 命令，
        让底层一次性连续开到目标交叉口，中途不作多余停留。
        """
        if self.pos_x != 0:
            raise RuntimeError("当前不在主干道中心，无法进行连续纵向移动！")
            
        if target_y > self.pos_y:
            self.turn_to(NORTH)
        elif target_y < self.pos_y:
            self.turn_to(SOUTH)
        else:
            return # 已经在目标位置
            
        # 根据目标交叉口下发对应的连续直达指令 (例如 TF1, TF2, TF3, TF4)
        cmd_str = f"TF {target_y}"
        self.execute_cmd(cmd_str)
        self.pos_y = target_y

    def wait_for_signal(self, valid_signals, timeout=None):
        print(f"\n[等待指令] 正在监听控制信号 {valid_signals} (输入 S 可随时紧急停车) ...")
        
        if timeout is not None:
            print(f"    [超时设置] {timeout} 秒内未收到有效信号将自动继续。")
            start_time = time.time()

        while True:
            if stop_event.is_set(): raise RuntimeError("检测到紧急停车指令 S")
            
            if not self.hw.simulation_mode and self.hw.ser and self.hw.ser.in_waiting > 0:
                try:
                    line = self.hw.ser.readline().decode('utf-8').strip()
                    if line == "BTN":
                        print(f"\n🔘 [硬件按键] 收到 Arduino 物理按键触发！正在通过网络告知电脑端...")
                        send_status_to_pc("button_pressed")
                    # 【新增】捕获底层发来的障碍物急停信号
                    elif line == "Obstacle Stop":
                        print("\n🛑 [底层安全警报] 硬件触发全局障碍物急停！正在同步中断 Python 业务流...")
                        stop_event.set()
                        raise RuntimeError("底层硬件检测到障碍物，自动触发急停")
                except Exception:
                    pass
            
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

    def detect_and_handle_student(self, branch_x, current_y):
        """执行完整送物闭环：使用连续性行驶指令完成返回老师与重返学生的流程"""
        print(f" -> [流程触发] 启动超声波循迹探路...")
        self.go_detect_obstacle()
        self.pos_x = branch_x
        send_status_to_pc("arrived_student")
        
        self.wait_for_signal(["go_teacher"])
        
        # 1. 退出分支，回到分叉点
        print(" -> [退回分叉点] 线上180度掉头并返回交叉口...")
        self.execute_cmd('PN') 
        self.heading = (self.heading + 2) % 4 
        self.execute_cmd('TF') 
        self.pos_x = 0
        
        # 2. 【连续行驶】直接从当前交叉口连续开回起点找老师
        print(f" -> [连续返回起点] 正在从交叉口 {current_y} 连续开往真正的起点...")
        self.turn_to(SOUTH) # 朝向起点方向
        self.execute_cmd(f"TF {current_y}")
        self.pos_y = 0
        self.turn_to(WEST) # 面朝西侧(起点老师位置)
        self.execute_cmd('TF') # 驶入起点
        self.pos_x = -1
        send_status_to_pc("arrived_teacher")
        
        self.wait_for_signal(["return_student"])
        
        # 3. 【连续行驶】从起点出发，直接连续开回目标现场路口
        print(" -> [连续重返现场] 正在从起点连续开回主路并直达交叉口...")
        self.execute_cmd('PU') # 掉头，重新面朝东
        self.heading = EAST
        self.execute_cmd('TF') # 回到主路入口 (y=0)
        self.pos_x = 0
        
        # 使用连续行驶指令直接开到目标交叉口 current_y
        self.continuous_move_to_y(current_y) 
        
        target_heading = WEST if branch_x == -1 else EAST
        self.turn_to(target_heading) 
        
        print(" -> [前行至现场] 重新驶入侧边栏靠近障碍物点...")
        self.go_detect_obstacle()
        self.pos_x = branch_x
        send_status_to_pc("arrived_student")
        
        self.wait_for_signal(["return_patrol"])
        
        # 4. 退出分叉点，姿态补偿
        print(" -> [姿态补偿] 线上180度掉头并退回交叉口...")
        self.execute_cmd('PN')
        self.heading = (self.heading + 2) % 4
        self.execute_cmd('TF')
        self.pos_x = 0
        
        send_status_to_pc("route_rejoined")

    def perform_double_side_detection(self, current_y):
        print(f"\n -> 执行左侧(西侧)探测 (交叉口 {current_y})")
        self.turn_to(WEST)
        send_status_to_pc("intersection_reached")
        
        sig = self.wait_for_signal(["approach_student"], timeout=30)
        if sig == "approach_student":
            self.detect_and_handle_student(branch_x=-1, current_y=current_y)
            print(" -> [左侧] 完结。准备进入右侧探测...")
        else:
            print(" -> [左侧] 探测超时/无信号。准备进入右侧探测...")
            
        print(f"\n -> 执行右侧(东侧)探测 (交叉口 {current_y})")
        self.turn_to(EAST)
        send_status_to_pc("intersection_reached")
        
        sig_r = self.wait_for_signal(["approach_student"], timeout=30)
        if sig_r == "approach_student":
            self.detect_and_handle_student(branch_x=1, current_y=current_y)
            print(" -> [右侧] 完结。")
        else:
            print(" -> [右侧] 探测超时/无信号。")

# ==============================================================================
# 第三部分：主循环业务流 
# ==============================================================================
    def run(self):
        print(">>>>>>>>>> 算法启动：基于绝对坐标系与连续指令引导路线 <<<<<<<<<<\n")
        
        print("--- [初始化] 离开起点，前往主干道入口 ---")
        self.execute_cmd('TF')
        self.pos_x = 0 
        
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n==================== 第 {loop_count} 轮 循迹双向扫描循环 ====================")
            
            # --- 阶段 1：正向推进扫描 (1 -> 4) ---
            print("\n➡️ >>>>>> [正向巡逻阶段] (交叉口 1 -> 4) >>>>>>")
            for y in [1, 2, 3, 4]:
                print(f"\n--- 移动至交叉口 {y} ---")
                self.move_to_y(y)
                self.perform_double_side_detection(y)
                
            # --- 阶段 2：逆向倒序扫描 (3 -> 1) ---
            print("\n⬅️ <<<<<< [逆向巡逻阶段] (交叉口 3 -> 1) <<<<<<")
            for y in [3, 2, 1]:
                print(f"\n--- 返回至交叉口 {y} ---")
                self.move_to_y(y)
                self.perform_double_side_detection(y)
                
            print("\n🏁 [本轮结束] 已完成全路段双向探测，即将无缝衔接下一轮巡逻！")


if __name__ == "__main__":
    # 【新增】1. 硬件连接和后台监听线程只在程序最开始启动一次！
    hw = CarHardware(baudrate=9600)
    
    listener_thread = threading.Thread(target=background_input_listener, args=(hw,), daemon=True)
    listener_thread.start()

    tcp_thread = threading.Thread(target=tcp_server_listener, args=(hw,), daemon=True)
    tcp_thread.start()
    
    # 【新增】2. 外层增加主循环，用于被 S 打断后重新回到等待状态
    while True:
        # 重置关键变量，防止上一次的 S 标志或废弃指令影响下一次运行
        stop_event.clear()
        while not signal_queue.empty():
            signal_queue.get_nowait()
            
        print("\n" + "="*50)

        # 【新增】在此处拦截！如果网络还没连上，主线程就在这死等，不打印输入框
        if not network_ready_event.is_set():
            print("⏳ 正在等待电脑端视觉程序接入网络...")
            network_ready_event.wait()  # 阻塞，直到 tcp_server_listener 执行了 .set()
            time.sleep(0.5) # 稍微睡 0.5 秒，让后台的 "✅已连接" 日志先打印完，避免和输入框抢位置

        print("👉 请在终端输入 'start' 并回车以开始整个完整的过程:")

        while True:
            try:
                # 【修改】统一从队列中获取输入内容，完美避免线程抢夺
                start_cmd = signal_queue.get(timeout=0.2)
                
                if start_cmd.lower() == 'start':
                    print("▶️ [启动成功] 系统控制权限已激活，流程正式开始！")
                    print("💡 [提示] 在终端随时敲击 'S' 并回车可命令小车紧急刹车并退出当前循环。\n")
                    break
                else:
                    print(f"[错误] 指令 '{start_cmd}' 未识别，必须先输入 'start'。")
            except queue.Empty:
                # 队列里没东西就一直循环等待，不卡死主程序
                continue

        # 每次重新 start 时，都重新实例化控制器，确保路径的 current_idx 等状态完全归零
        car = CarController(hw)
        
        try:
            car.run()
        except RuntimeError as e:
            # 捕获到由于按 S 触发的 RuntimeError
            print(f"\n🛑 [业务流中断] {e}")
            print("[系统提示] 小车已安全制动。正在重置业务流程...\n")
            # 【修改】不要调用 sys.exit(0)，让程序自然走到这一步结束，然后借由外层 while True 重回输入 start
        except KeyboardInterrupt:
            # 只有按 Ctrl+C 才会真正退出程序
            print("\n[系统提示] 检测到 Ctrl+C，强制退出程序。")
            sys.exit(0)