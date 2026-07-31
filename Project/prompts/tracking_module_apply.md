## Date: 23 July, 2026

---

### 需求
编写利用寻迹模块实现小车按地上黑色线轨迹前进、后退、转向。

### 交付要求
一个 `.ino` 程序，在 Arduino 板载程序中通过新的串口命令解析映射关系，实现 [需求] 中的功能。

### 部署位置
代码部署在 Arduino 主板上。

### 硬件资源
- 6 个 HW-511 循迹模块，其中 0, 1 号装于小车前端，2, 3 号装于小车中部两侧，4, 5 号装于小车后端。
- 6 个 HW-511 循迹模块的信号输出接口尚未确定，在程序中将这些接口悬空，并指明模块编号与其在小车上物理位置的对应关系。

### 参考程序
- `car_control.ino` 文件是现有的 Arduino 控制程序，通过树莓派接收串口数据，解析为对应的运动命令。

### 程序实现详细要求
1. 利用小车前端、中部的循迹模块，实现小车按地面黑色直线前进直到交叉口中央。
2. 利用小车前端、中部的循迹模块，实现小车按地面黑色直角弯线（只有直角黑线，没有其它方向上的黑线）做直角左右转弯。
3. 利用小车后端、中部的循迹模块，实现小车按地面黑色直线后退直到交叉口中央。
4. 利用小车中部的循迹模块，实现小车按地面交叉口向左、向右转向 90 度。
5. 综合分析上述任务需求，整合各类任务需求，编写完整的 `.ino` 代码，满足上述要求。

---

### 角色设定
你是一位经验丰富的嵌入式软件工程师与机器人开发者。请根据以下需求，为基于 Arduino 与 `AFMotor` 驱动的机器人小车编写/修改 C++ (`.ino`) 控制程序。

### 任务目标
在现有的串口控制小车程序基础上，扩展基于 HW-511 模块的循迹功能。通过增加新的串口指令，使小车能够自主完成沿地面黑线的前进、后退、直角转向及交叉路口转向。

### 硬件拓扑与资源
1. **电机驱动**：使用 `AFMotor.h` 库控制 4 个直流电机（`motor1` 至 `motor4`），依靠四轮差速实现运动与转向。
2. **循迹模块**：共部署 6 个 HW-511 循迹模块，具体物理位置映射如下：
   - **前端**：0 号（左前）、1 号（右前）。
   - **中部**：2 号（左中）、3 号（右中）。
   - **后端**：4 号（左后）、5 号（右后）。
3. **引脚分配**：6 个 HW-511 模块的信号输出引脚目前**尚未确定**。请在代码顶部使用 `#define` 或 `const int` 显式声明这 6 个引脚变量，暂时赋值为悬空或占位符（如 `A0`-`A5` 或 `-1`），并务必在注释中清晰标明“模块编号 - 物理位置 - 引脚变量”的对应关系。

### 核心功能需求
需要在程序中新增对应的逻辑函数，通过解析新的串口命令（请自行定义合适的字符指令，例如 `TF` 代表循迹前进）触发以下动作：

1. **循迹直行（至交叉口）**：利用前端（0, 1号）和中部（2, 3号）模块，使小车沿黑色直线前进。当检测到到达交叉口中央（如多个传感器同时触发黑线特征）时，自动停止并向串口返回完成信号。
2. **直角弯道转向**：在只有直角黑线（无其它方向黑线）的路口，利用前端和中部模块，使小车沿黑线完成 90 度的自动左转或右转。
3. **循迹后退（至交叉口）**：利用后端（4, 5号）和中部（2, 3号）模块，使小车沿黑色直线倒车后退。当检测到退至交叉口中央时，自动停止并返回完成信号。
4. **路口原地转向**：在标准的交叉路口，仅依靠中部（2, 3号）模块作为基准，使小车在路口中心向左或向右原地旋转 90 度，对准新方向的黑线后停止并返回完成信号。

### 现有代码集成要求
1. **非阻塞与兼容性**：请将循迹逻辑与现有的 `car_control.ino` 框架无缝整合。在循迹循环中，必须保留并调用 `checkEmergencyStop()` 函数，确保在自动循迹过程中随时可以通过串口发送 `S` 命令触发急停。
2. **复用底层控制**：复用现有的电机驱动对象和运动控制风格，确保基础的加减速和转向逻辑与现有代码兼容。

### 交付标准
1. 输出一份完整、可编译的 `.ino` 代码，包含原有的功能和新增的循迹功能。
2. 代码需具备清晰的中文注释，特别是循迹状态判断的条件语句，需解释清楚各个传感器的状态组合意图（如 `HIGH`/`LOW` 代表黑线或白板）。
3. 在 `setup()` 的串口打印提示中，补充新增循迹命令的说明。

### 参考程序（现有 car_control.ino）
```cpp
#include <AFMotor.h>

// ---------- 电机对象 ----------
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// ---------- 超声波引脚 ----------
const int TRIGGER = 22;
const int ECHO = 24;

// ---------- 按键引脚 -----------
const int BUTTON_PIN = 26;

// ---------- 编码器引脚（两个后轮，Arduino Mega）----------
const int ENC_PIN_RIGHT = 18;   // 右后轮编码器
const int ENC_PIN_LEFT  = 19;   // 左后轮编码器
volatile long encoderCountLeft = 0;   // 左后轮脉冲计数
volatile long encoderCountRight = 0;  // 右后轮脉冲计数

// ---------- 运动参数 ----------
int speedValue = 200;                // 行驶速度（0~255）
const int OBSTACLE_DIST = 30;        // 障碍物距离阈值（cm）

// 编码器校准（请根据实际硬件修改）
const float CM_PER_PULSE = 20.0 / 4.0; // 每个脉冲对应 5.0 cm
const float WHEEL_BASE = 13.0;         // 左右轮中心距（cm），需实测

// ---------- 角度缩放系数（固定为2.0） ----------
float angleScale = 2.0;   // 硬编码，不可动态修改

// ---------- 串口命令存储 ----------
String command = "";

bool checkEmergencyStop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "S") {
      stopAllMotors();
      Serial.println("Emergency Stop");
      return true;
    }
  }
  return false;
}

void setup() {
  Serial.begin(9600);
  Serial.println("Car Ready! Commands:");
  Serial.println("  F/B/L/R/S - Manual control");
  Serial.println("  O - Drive forward until obstacle, then send 'D'");
  Serial.println("  G [dist] - Move distance (cm), positive forward, negative backward");
  Serial.println("  T [angle] - Turn angle (degrees), positive left, negative right");
  Serial.println("  (After G/T completes, 'Done' is sent)");
  Serial.println("  RST - Reset encoder counts");
  Serial.print("Current angle scale (fixed): ");
  Serial.println(angleScale);

  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  stopAllMotors();

  pinMode(TRIGGER, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(ENC_PIN_LEFT, INPUT_PULLUP);
  pinMode(ENC_PIN_RIGHT, INPUT_PULLUP);

  // 中断：上升沿触发
  attachInterrupt(digitalPinToInterrupt(ENC_PIN_LEFT), encoderLeftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN_RIGHT), encoderRightISR, RISING);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void loop() {
  // --- 新增：按键监听与消抖逻辑 ---
  static int lastButtonState = HIGH;
  static unsigned long lastDebounceTime = 0;
  
  int reading = digitalRead(BUTTON_PIN);
  
  // 如果状态发生变化，重置消抖计时器
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  // 只有当状态稳定超过 50ms 时，才认为是一次真实的按压
  if ((millis() - lastDebounceTime) > 50) {
    static int buttonState = HIGH;
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == LOW) {
        // 确认被按下，向 Python 发送暗号
        Serial.println("BTN");
      }
    }
  }
  lastButtonState = reading;
  
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() == 0) return;

    char cmd = command.charAt(0);
    String param = command.substring(1);
    param.trim();

    if (cmd == 'F') forward();
    else if (cmd == 'B') backward();
    else if (cmd == 'L') turnLeft();
    else if (cmd == 'R' && param.length() == 0) turnRight();
    else if (cmd == 'S') stopAllMotors();
    else if (cmd == 'O') driveUntilObstacle();
    else if (cmd == 'G') {
      float dist = param.toFloat();
      if (dist == 0 && param != "0") {
        Serial.println("Invalid distance");
        return;
      }
      moveDistance(dist);
    }
    else if (cmd == 'T') {
      float angle = param.toFloat();
      if (angle == 0 && param != "0") {
        Serial.println("Invalid angle");
        return;
      }
      turnAngle(angle);
    }
    else if (cmd == 'R' && param == "ST") { // "RST"
      noInterrupts();
      encoderCountLeft = 0;
      encoderCountRight = 0;
      interrupts();
      Serial.println("Encoders reset");
    }
    else {
      Serial.println("Unknown command");
    }
    return;
  }
}

// ========== 编码器中断服务 ==========
void encoderLeftISR() { encoderCountLeft++; }
void encoderRightISR() { encoderCountRight++; }

// ========== 直线移动（距离 cm） ==========
void moveDistance(float dist_cm) { /* 省略细节，保留原有逻辑 */ }

// ========== 原地旋转（角度度） ==========
void turnAngle(float angle_deg) { /* 省略细节，保留原有逻辑 */ }

// ========== 直行直到障碍物 ==========
void driveUntilObstacle() { /* 省略细节，保留原有逻辑 */ }

// ========== 测距函数 ==========
float measureDistance() { /* 省略细节，保留原有逻辑 */ }

// ========== 电机控制函数 ==========
void forward() { motor1.run(FORWARD); motor2.run(FORWARD); motor3.run(FORWARD); motor4.run(FORWARD); Serial.println("Forward"); }
void backward() { motor1.run(BACKWARD); motor2.run(BACKWARD); motor3.run(BACKWARD); motor4.run(BACKWARD); Serial.println("Backward"); }
void turnLeft() { motor1.run(BACKWARD); motor2.run(FORWARD); motor3.run(FORWARD); motor4.run(BACKWARD); Serial.println("Turn Left"); }
void turnRight() { motor1.run(FORWARD); motor2.run(BACKWARD); motor3.run(BACKWARD); motor4.run(FORWARD); Serial.println("Turn Right"); }
void stopAllMotors() { motor1.run(RELEASE); motor2.run(RELEASE); motor3.run(RELEASE); motor4.run(RELEASE); Serial.println("Stopped"); }
\```
