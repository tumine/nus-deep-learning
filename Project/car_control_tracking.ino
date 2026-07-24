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

// ---------- 全局障碍物守护 ----------
const unsigned long OBSTACLE_CHECK_INTERVAL = 60; // 障碍物检测最小间隔（ms），限频避免阻塞
unsigned long lastObstacleCheck = 0;              // 上次障碍物检测时刻
bool manualMoving = false;                        // 手动命令(F/B/L/R)运动中标志
bool manualForward = false;                       // 手动运动是否为前进（仅前进受障碍守护）

// 编码器校准（请根据实际硬件修改）
const float CM_PER_PULSE = 20.0 / 4.0; // 每个脉冲对应 5.0 cm
const float WHEEL_BASE = 13.0;         // 左右轮中心距（cm），需实测

// ---------- 角度缩放系数（固定为2.0） ----------
float angleScale = 2.0;   // 硬编码，不可动态修改

// ---------- HW-511 循迹模块引脚（引脚尚未确定，先用 A0-A5 占位，接线后请修改） ----------
// 模块编号 - 物理位置 - 引脚变量
const int TRACK_PIN_FL = 39;   // 0号 - 左前  (Front Left)
const int TRACK_PIN_FR = 41;   // 1号 - 右前  (Front Right)
const int TRACK_PIN_ML = 37;   // 2号 - 左中  (Middle Left)
const int TRACK_PIN_MR = 43;   // 3号 - 右中  (Middle Right)
const int TRACK_PIN_RL = 35;   // 4号 - 左后  (Rear Left)
const int TRACK_PIN_RR = 45;   // 5号 - 右后  (Rear Right)

// HW-511 输出电平约定：压在黑线上（不反光）输出 HIGH，白色地面（反光）输出 LOW。
// 若实际模块极性相反，只需把下面的宏改为 LOW。
#define ON_LINE HIGH

// ---------- 循迹运动参数 ----------
const int TRACK_SPEED       = 160;    // 循迹直行基础速度（0~255）
const int TRACK_SLOW_SPEED  = 60;     // 循迹修正时内侧减速后的速度
const int TRACK_TURN_SPEED  = 150;    // 原地旋转/弯道转向速度
const unsigned long TRACK_TIMEOUT = 30000; // 循迹动作超时（ms）
const unsigned long TRACK_TURN_MINTIME1 = 150; // 转向时间阈值（进入白区），超过该阈值后才可触发停下
const unsigned long TRACK_TURN_MINTIME2 = 250; // 转向时间阈值（再次检测到黑区），超过该阈值后才可触发停下

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

// ---------- 全局障碍物守护：限频测距，30cm 内立即停车 ----------
// 注意：守护仅对前进类运动生效（F / G正值 / TF），
// 后退与转向不受影响，否则障碍物仍在 30cm 内时小车无法脱困。
// 返回 true 表示已因障碍物停车，调用方应中止当前动作
bool checkObstacle() {
  unsigned long now = millis();
  if (now - lastObstacleCheck < OBSTACLE_CHECK_INTERVAL) return false;
  lastObstacleCheck = now;

  float cm = measureDistanceFast();
  if (cm > 0 && cm < OBSTACLE_DIST) {
    stopAllMotors();
    Serial.println("Obstacle Stop");
    return true;
  }
  return false;
}

// ---------- 组合检查：急停命令 或 障碍物，任一触发即中止 ----------
bool checkStopConditions() {
  if (checkEmergencyStop()) return true;
  if (checkObstacle()) return true;
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
  Serial.println("  --- Line tracking (HW-511) ---");
  Serial.println("  TF [n] - Track forward until nth intersection (default 1)");
  Serial.println("  TB [n] - Track backward until nth intersection (default 1)");
  Serial.println("  CL/CR - Follow 90-deg corner, turn left/right");
  Serial.println("  PL/PR - Pivot 90-deg left/right at intersection");
  Serial.println("  PU - Pivot 180-deg (U-turn) on the intersection");
  Serial.println("  PN - Pivot 180-deg (U-turn) on the line");
  Serial.println("  TO - Track forward until obstacle, then send 'D:<dist>'");
  Serial.println("  (After tracking completes, 'Done' is sent)");
  Serial.println("  Ultrasonic guard: forward motion auto-stops when obstacle < 30cm ('Obstacle Stop')");
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

  // 循迹模块信号输入
  pinMode(TRACK_PIN_FL, INPUT);
  pinMode(TRACK_PIN_FR, INPUT);
  pinMode(TRACK_PIN_ML, INPUT);
  pinMode(TRACK_PIN_MR, INPUT);
  pinMode(TRACK_PIN_RL, INPUT);
  pinMode(TRACK_PIN_RR, INPUT);

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

  // --- 全局障碍物守护：仅手动前进(F)期间持续检测，后退/转向不受限以便脱困 ---
  if (manualMoving && manualForward) {
    checkObstacle();  // 触发时内部会 stopAllMotors() 并清除 manualMoving
  }

  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() == 0) return;

    // ---------- 循迹类多字符命令（必须先于单字符解析，避免 "TF" 被 'T' 吞掉） ----------
    if (command == "TF" || command.startsWith("TF ")) {
      int n = parseIntersectionCount(command.substring(2));
      if (n <= 0) { Serial.println("Invalid intersection count"); return; }
      trackForward(n);
      return;
    }
    else if (command == "TB" || command.startsWith("TB ")) {
      int n = parseIntersectionCount(command.substring(2));
      if (n <= 0) { Serial.println("Invalid intersection count"); return; }
      trackBackward(n);
      return;
    }
    else if (command == "TO") { trackForwardUntilObstacle(); return; }
    else if (command == "CL") { cornerTurn(true);   return; }
    else if (command == "CR") { cornerTurn(false);  return; }
    else if (command == "PL") { pivotLeft();        return; }
    else if (command == "PR") { pivotRight();       return; }
    else if (command == "PU") { pivotUTurn();       return; }
    else if (command == "PN") { pivotLeft();        return; }

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
    // ---------- C 命令已移除 ----------
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
void encoderLeftISR() {
  encoderCountLeft++;
}
void encoderRightISR() {
  encoderCountRight++;
}

// ========== 直线移动（距离 cm） ==========
void moveDistance(float dist_cm) {
  long targetPulses = (long)(dist_cm / CM_PER_PULSE);
  if (targetPulses == 0) {
    Serial.println("Distance too small");
    return;
  }

  noInterrupts();
  encoderCountLeft = 0;
  encoderCountRight = 0;
  interrupts();

  long needed = abs(targetPulses);
  long avgPulses = 0;

  bool movingForward = (targetPulses > 0);
  if (movingForward) forward(); else backward();

  unsigned long start = millis();
  unsigned long lastPrintTime = start;   // 用于控制打印频率

  while (avgPulses < needed) {
    // 障碍守护仅对前进生效；后退只响应急停命令
    if (movingForward ? checkStopConditions() : checkEmergencyStop()) {
        return;
    }
    
    noInterrupts();
    long left = abs(encoderCountLeft);
    long right = abs(encoderCountRight);
    interrupts();
    avgPulses = (left + right) / 2;

    if (millis() - lastPrintTime >= 100) {
      Serial.print("L:");
      Serial.print(left);
      Serial.print(" R:");
      Serial.print(right);
      Serial.print(" Avg:");
      Serial.println(avgPulses);
      lastPrintTime = millis();
    }

    if (millis() - start > 30000) {
      stopAllMotors();
      Serial.println("Timeout");
      return;
    }
    delay(1);
  }
  stopAllMotors();

  Serial.print("Final L:");
  Serial.print(abs(encoderCountLeft));
  Serial.print(" R:");
  Serial.print(abs(encoderCountRight));
  Serial.print(" Avg:");
  Serial.println((abs(encoderCountLeft) + abs(encoderCountRight)) / 2);
  Serial.println("Done");
}

// ========== 原地旋转（角度度），应用固定缩放系数 ==========
void turnAngle(float angle_deg) {
  // 应用固定的缩放系数
  float scaledAngle = angle_deg * angleScale;

  // 计算每个轮子需要移动的距离 s = (角度弧度 * 轮距) / 2
  float angle_rad = scaledAngle * 3.14159265 / 180.0;
  float s = abs(angle_rad) * WHEEL_BASE / 2.0;
  long targetPulses = (long)(s / CM_PER_PULSE);
  if (targetPulses == 0) {
    Serial.println("Angle too small");
    return;
  }

  noInterrupts();
  encoderCountLeft = 0;
  encoderCountRight = 0;
  interrupts();

  // 设置旋转方向（正角度 → 左转）
  if (scaledAngle > 0) {
    // 左转：左轮后退，右轮前进
    motor1.run(BACKWARD);
    motor2.run(FORWARD);
    motor3.run(FORWARD);
    motor4.run(BACKWARD);
  } else {
    // 右转：左轮前进，右轮后退
    motor1.run(FORWARD);
    motor2.run(BACKWARD);
    motor3.run(BACKWARD);
    motor4.run(FORWARD);
  }

  long needed = targetPulses;
  long avgPulses = 0;
  unsigned long start = millis();
  unsigned long lastPrintTime = start;

  while (avgPulses < needed) {

    // 原地旋转不受障碍守护限制（否则障碍物前无法转向脱困）
    if (checkEmergencyStop()) {
        return;
    }

    noInterrupts();
    long left = abs(encoderCountLeft);
    long right = abs(encoderCountRight);
    interrupts();
    avgPulses = (left + right) / 2;

    if (millis() - lastPrintTime >= 100) {
      Serial.print("L:");
      Serial.print(left);
      Serial.print(" R:");
      Serial.print(right);
      Serial.print(" Avg:");
      Serial.println(avgPulses);
      lastPrintTime = millis();
    }

    if (millis() - start > 30000) {
      stopAllMotors();
      Serial.println("Timeout");
      return;
    }
    delay(1);
  }
  stopAllMotors();

  Serial.print("Final L:");
  Serial.print(abs(encoderCountLeft));
  Serial.print(" R:");
  Serial.print(abs(encoderCountRight));
  Serial.print(" Avg:");
  Serial.println((abs(encoderCountLeft) + abs(encoderCountRight)) / 2);
  Serial.println("Done");
}

// ========== 直行直到障碍物 ==========
void driveUntilObstacle() {
  // 1. 开始探测前，先清零编码器
  noInterrupts();
  encoderCountLeft = 0;
  encoderCountRight = 0;
  interrupts();

  while (true) {
    if (checkEmergencyStop()) {
        return;
    }
    float cm = measureDistance();
    if (cm > 0 && cm < OBSTACLE_DIST) {
      stopAllMotors();
      
      // 2. 停车后，计算走过的距离
      noInterrupts();
      long left = abs(encoderCountLeft);
      long right = abs(encoderCountRight);
      interrupts();
      long avgPulses = (left + right) / 2;
      float traveled_cm = avgPulses * CM_PER_PULSE;
      
      // 3. 将距离传回给 Python (格式如: D:45.5)
      Serial.print("D:");
      Serial.println(traveled_cm);
      break;
    }
    forward();
    delay(50);
  }
}

// ========== 测距函数 ==========
float measureDistance() {
  digitalWrite(TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER, LOW);
  unsigned long duration = pulseIn(ECHO, HIGH, 30000);
  return duration / 58.2;
}

// ========== 快速测距（守护专用，短超时避免阻塞循迹循环） ==========
// 超时 5000us ≈ 最远约 86cm，足以判断 30cm 阈值；无回波返回 0（视为无障碍）
float measureDistanceFast() {
  digitalWrite(TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER, LOW);
  unsigned long duration = pulseIn(ECHO, HIGH, 5000);
  return duration / 58.2;
}

// ========== 电机控制函数 ==========
void forward() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  manualMoving = true;
  manualForward = true;
  Serial.println("Forward");
}

void backward() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  manualMoving = true;
  manualForward = false;
  Serial.println("Backward");
}

void turnLeft() {
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
  manualMoving = true;
  manualForward = false;
  Serial.println("Turn Left");
}

void turnRight() {
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
  manualMoving = true;
  manualForward = false;
  Serial.println("Turn Right");
}

void stopAllMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
  manualMoving = false;
  manualForward = false;
  Serial.println("Stopped");
}

// ==========================================================
// ================ HW-511 循迹功能实现 =====================
// ==========================================================
// 电机布局约定（与 turnLeft/turnRight 一致）：
//   motor1 / motor4 = 左侧轮，motor2 / motor3 = 右侧轮

// ---------- 传感器读取：true = 压在黑线上 ----------
bool onLine(int pin) {
  return digitalRead(pin) == ON_LINE;
}

// ---------- 设置左右两侧速度（差速修正用） ----------
void setSideSpeed(int leftSpd, int rightSpd) {
  motor1.setSpeed(leftSpd);
  motor4.setSpeed(leftSpd);
  motor2.setSpeed(rightSpd);
  motor3.setSpeed(rightSpd);
}

// ---------- 恢复默认速度（循迹结束后必须调用） ----------
void restoreSpeed() {
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
}

// ---------- 无串口打印的底层运动（循迹循环内高频调用，避免刷屏） ----------
void driveForwardSilent() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
}

void driveBackwardSilent() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
}

void rotateLeftSilent() {   // 左转：左轮后退，右轮前进
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
}

void rotateRightSilent() {  // 右转：左轮前进，右轮后退
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
}

void stopSilent() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

// ---------- 循迹动作统一收尾 ----------
void trackFinish(const char* msg) {
  stopSilent();
  restoreSpeed();
  Serial.println(msg);
}

// ---------- 解析 TF/TB 的可选路口数参数：空串默认 1，非法返回 -1 ----------
int parseIntersectionCount(String param) {
  param.trim();
  if (param.length() == 0) return 1;
  int n = param.toInt();
  if (n <= 0) return -1;
  return n;
}

// ==========================================================
// TF [n]：循迹直行（前进），直到到达连续第 n 个交叉路口中央
// 使用前端（0,1号）做方向修正，中部（2,3号）判定交叉口
// ==========================================================
void trackForward(int nStop) {
  unsigned long start = millis();
  int crossCount = 0;

  while (true) {
    if (checkStopConditions()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }

    bool fl = onLine(TRACK_PIN_FL);  // 左前
    bool fr = onLine(TRACK_PIN_FR);  // 右前
    bool ml = onLine(TRACK_PIN_ML);  // 左中
    bool mr = onLine(TRACK_PIN_MR);  // 右中

    // 【交叉口判定】中部左右两个传感器同时压黑：
    // 说明横向黑线已经到达车身中部，车体中心位于路口中央
    if (ml && mr) {
      crossCount++;
      if (crossCount >= nStop) {
        // 【关键数据】补偿运动的时间
        delay(60);  // 满电量时数据
        // delay(80);  // 中等电量时数据
        // 低电量时数据（待测）
        trackFinish("Done");
        return;
      }
      // 未到目标路口：直行穿过横线，直到中部传感器离开黑线，避免重复计数
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
      while (onLine(TRACK_PIN_ML) || onLine(TRACK_PIN_MR)) {
        if (checkStopConditions()) { restoreSpeed(); return; }
        if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
        delay(2);
      }
      continue;
    }

    // 【方向修正】理想状态：黑线位于两前端传感器之间，0/1号都在白色区域
    if (fl && fr) {
      // 两前端同时压黑：多为提前碰到路口横线，继续直行交给中部确认
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
    } else if (fl) {
      // 仅左前压黑（HIGH）：黑线偏向车体左侧 → 左侧减速向左修正
      setSideSpeed(TRACK_SLOW_SPEED, TRACK_SPEED + 60);
      driveForwardSilent();
    } else if (fr) {
      // 仅右前压黑：黑线偏向车体右侧 → 右侧减速向右修正
      setSideSpeed(TRACK_SPEED + 60, TRACK_SLOW_SPEED);
      driveForwardSilent();
    } else {
      // 两前端都在白色区域：车体居中 → 全速直行
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
    }
    delay(5);
  }
}

// ==========================================================
// TB [n]：循迹倒车（后退），直到退至连续第 n 个交叉路口中央
// 使用后端（4,5号）做方向修正，中部（2,3号）判定交叉口
// ==========================================================
void trackBackward(int nStop) {
  unsigned long start = millis();
  int crossCount = 0;

  while (true) {
    // 倒车不受障碍守护限制（超声波朝前，且需允许退离障碍）
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }

    bool rl = onLine(TRACK_PIN_RL);  // 左后
    bool rr = onLine(TRACK_PIN_RR);  // 右后
    bool ml = onLine(TRACK_PIN_ML);  // 左中
    bool mr = onLine(TRACK_PIN_MR);  // 右中

    // 【交叉口判定】中部左右同时压黑 → 车体中心退到路口中央
    if (ml && mr) {
      crossCount++;
      if (crossCount >= nStop) {
        trackFinish("Done");
        return;
      }
      // 未到目标路口：继续后退穿过横线，直到中部传感器离开黑线，避免重复计数
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveBackwardSilent();
      while (onLine(TRACK_PIN_ML) || onLine(TRACK_PIN_MR)) {
        if (checkEmergencyStop()) { restoreSpeed(); return; }
        if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
        delay(2);
      }
      continue;
    }

    // 【方向修正】倒车时以后端传感器为准：
    // 后退中车尾向哪边偏，对应侧后端传感器就会压到黑线
    if (rl && rr) {
      // 两后端同时压黑：多为碰到路口横线，继续后退交给中部确认
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      delay(50);
      driveBackwardSilent();
    } else if (rl) {
      // 仅左后压黑：黑线偏向车尾左侧 → 左侧减速，让车尾向左靠回黑线
      setSideSpeed(TRACK_SLOW_SPEED, TRACK_SPEED + 40);
      delay(50);
      driveBackwardSilent();
    } else if (rr) {
      // 仅右后压黑：黑线偏向车尾右侧 → 右侧减速修正
      setSideSpeed(TRACK_SPEED + 40, TRACK_SLOW_SPEED);
      delay(50);
      driveBackwardSilent();
    } else {
      // 两后端都在白色区域：车尾居中 → 全速后退
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      delay(50);
      driveBackwardSilent();
    }
    delay(5);
  }
}

// ==========================================================
// CL / CR：直角弯道自动转向（沿黑线过 90 度弯）
// 场景：直线尽头只有一条直角黑线（无其它分支）
// 策略（三阶段）：
//   阶段1：原地旋转，直到前端两个传感器都离开原来的黑线（全白）
//   阶段2：继续旋转，直到内侧前端传感器重新压到新方向的黑线
//   阶段3：继续旋转，直到内侧前端传感器越过黑线回到白区，
//           此时黑线大致位于两前端传感器之间 → 停车
// ==========================================================
void cornerTurn(bool leftTurn) {
  unsigned long start = millis();
  setSideSpeed(TRACK_TURN_SPEED, TRACK_TURN_SPEED);

  // 内侧前端传感器：左转看左前（0号），右转看右前（1号）
  int innerFrontPin = leftTurn ? TRACK_PIN_FL : TRACK_PIN_FR;

  if (leftTurn) rotateLeftSilent();
  else          rotateRightSilent();

  // 转向过程不受障碍守护限制（需允许在障碍物前转向脱困）
  // ---- 阶段1：等待前端离开当前黑线（两前端均为白 LOW）----
  while (onLine(TRACK_PIN_FL) || onLine(TRACK_PIN_FR)) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  // ---- 阶段2：继续旋转，直到内侧前端重新压到新方向黑线（HIGH）----
  while (!onLine(innerFrontPin)) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  // ---- 阶段3：继续旋转少许，让内侧传感器越过黑线回到白区，
  //             黑线即落在两前端传感器之间，车头对准新方向 ----
  while (onLine(innerFrontPin)) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  trackFinish("Done");
}

// ==========================================================
// PL：交叉路口原地左旋 90 度
// 场景：车体中心已停在标准十字路口中央（通常由 TF/TB 停下）
// 以左前传感器（0号）为基准：
//   阶段1：原地左旋，直到左前传感器离开黑线进入白区（LOW）
//   阶段2：继续左旋，直到左前传感器再次压到黑线（HIGH），
//           说明车头已扫到新方向的黑线 → 立即停车
// ==========================================================
void pivotLeft() {
  unsigned long start = millis();
  setSideSpeed(TRACK_TURN_SPEED, TRACK_TURN_SPEED);

  rotateLeftSilent();

  // 原地旋转不受障碍守护限制
  // ---- 阶段1：等待左前传感器离开黑线，进入白区（LOW）----
  while (onLine(TRACK_PIN_FL) || millis() - start < TRACK_TURN_MINTIME1) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  // ---- 阶段2：继续左旋，直到左前传感器再次压到黑线（HIGH），
  //             即检测到新方向的黑线 → 停车 ----
  while (!onLine(TRACK_PIN_FL) || millis() - start < TRACK_TURN_MINTIME2) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  trackFinish("Done");
}

// ==========================================================
// TO：循迹直行（前进），直到超声波检测到障碍物（参考 O 命令）
// 方向修正逻辑与 TF 相同（前端 0/1 号传感器），
// 但不在交叉口停车，而是持续循迹直到前方出现障碍物。
// 停车后通过编码器计算行驶距离，发送 "D:<distance>"。
// ==========================================================
void trackForwardUntilObstacle() {
  unsigned long start = millis();

  // 开始前清零编码器（与 O 命令一致）
  noInterrupts();
  encoderCountLeft = 0;
  encoderCountRight = 0;
  interrupts();

  while (true) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }

    // 【障碍物判定】与 O 命令相同的距离阈值
    float cm = measureDistance();
    if (cm > 0 && cm < OBSTACLE_DIST) {
      stopSilent();
      restoreSpeed();

      // 计算走过的距离并回传（格式如: D:45.5）
      noInterrupts();
      long left = abs(encoderCountLeft);
      long right = abs(encoderCountRight);
      interrupts();
      long avgPulses = (left + right) / 2;
      float traveled_cm = avgPulses * CM_PER_PULSE;

      Serial.print("D:");
      Serial.println(traveled_cm);
      Serial.println("Done");
      return;
    }

    bool fl = onLine(TRACK_PIN_FL);  // 左前
    bool fr = onLine(TRACK_PIN_FR);  // 右前

    // 【方向修正】与 TF 相同，但交叉口横线不停车，直接穿过
    if (fl && fr) {
      // 两前端同时压黑：碰到路口横线 → 直行穿过
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
    } else if (fl) {
      // 仅左前压黑：黑线偏向车体左侧 → 左侧减速向左修正
      setSideSpeed(TRACK_SLOW_SPEED, TRACK_SPEED + 60);
      driveForwardSilent();
    } else if (fr) {
      // 仅右前压黑：黑线偏向车体右侧 → 右侧减速向右修正
      setSideSpeed(TRACK_SPEED + 60, TRACK_SLOW_SPEED);
      driveForwardSilent();
    } else {
      // 两前端都在白色区域：车体居中 → 全速直行
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
    }
    delay(5);
  }
}

// ==========================================================
// PR：交叉路口原地右旋 90 度
// 场景：车体中心已停在标准十字路口中央（通常由 TF/TB 停下）
// 以右前传感器（1号）为基准：
//   阶段1：原地右旋，直到右前传感器离开黑线进入白区（LOW）
//   阶段2：继续右旋，直到右前传感器再次压到黑线（HIGH），
//           说明车头已扫到新方向的黑线 → 立即停车
// ==========================================================
void pivotRight() {
  unsigned long start = millis();
  setSideSpeed(TRACK_TURN_SPEED, TRACK_TURN_SPEED);

  rotateRightSilent();

  // 原地旋转不受障碍守护限制
  // ---- 阶段1：等待右前传感器离开黑线，进入白区（LOW）----
  while (onLine(TRACK_PIN_FR) || millis() - start < TRACK_TURN_MINTIME1) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  // ---- 阶段2：继续右旋，直到右前传感器再次压到黑线（HIGH），
  //             即检测到新方向的黑线 → 停车 ----
  while (!onLine(TRACK_PIN_FR) || millis() - start < TRACK_TURN_MINTIME2) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  trackFinish("Done");
}

// ==========================================================
// PU：循迹原地旋转 180 度（掉头）
// 场景：车体中心停在标准十字路口中央（通常由 TF/TB 停下），
//       与 PL/PR 使用场景一致。
// 以左前传感器（0号）为基准，向左连续执行两段 90 度扫线
// （每段 = 离开黑线 → 再次压到黑线，与 PL 单段逻辑相同）：
//   第1段：离开原黑线 → 扫到垂直方向黑线（旋转约90度）
//   第2段：离开垂直黑线 → 扫到原黑线反向延长线（再旋转约90度）
// 每段均有最小旋转时间约束，避免误触发。
// 注意：若在无交叉线的直线段使用，传感器每约180度才扫线一次，
//       两段逻辑会旋转过头，请仅在十字路口使用本命令。
// ==========================================================
void pivotUTurn() {
  unsigned long start = millis();
  setSideSpeed(TRACK_TURN_SPEED, TRACK_TURN_SPEED);

  rotateLeftSilent();

  // 连续执行两段 90 度扫线（离线→压线），共 180 度
  for (int seg = 0; seg < 2; seg++) {
    unsigned long segStart = millis();

    // 原地旋转不受障碍守护限制
    // ---- 阶段1：等待左前传感器离开黑线，进入白区（LOW）----
    while (onLine(TRACK_PIN_FL) || millis() - segStart < TRACK_TURN_MINTIME1) {
      if (checkEmergencyStop()) { restoreSpeed(); return; }
      if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
      delay(2);
    }

    // ---- 阶段2：继续左旋，直到左前传感器再次压到黑线（HIGH）----
    while (!onLine(TRACK_PIN_FL) || millis() - segStart < TRACK_TURN_MINTIME2) {
      if (checkEmergencyStop()) { restoreSpeed(); return; }
      if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
      delay(2);
    }
  }

  trackFinish("Done");
}