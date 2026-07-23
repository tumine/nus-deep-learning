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
const int TRACK_SLOW_SPEED  = 90;     // 循迹修正时内侧减速后的速度
const int TRACK_TURN_SPEED  = 150;    // 原地旋转/弯道转向速度
const unsigned long TRACK_TIMEOUT = 30000; // 循迹动作超时（ms）

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
  Serial.println("  --- Line tracking (HW-511) ---");
  Serial.println("  TF - Track forward until intersection");
  Serial.println("  TB - Track backward until intersection");
  Serial.println("  CL/CR - Follow 90-deg corner, turn left/right");
  Serial.println("  PL/PR - Pivot 90-deg left/right at intersection");
  Serial.println("  (After tracking completes, 'Done' is sent)");
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
  
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() == 0) return;

    // ---------- 循迹类多字符命令（必须先于单字符解析，避免 "TF" 被 'T' 吞掉） ----------
    if (command == "TF")      { trackForward();     return; }
    else if (command == "TB") { trackBackward();    return; }
    else if (command == "CL") { cornerTurn(true);   return; }
    else if (command == "CR") { cornerTurn(false);  return; }
    else if (command == "PL") { pivotTurn(true);    return; }
    else if (command == "PR") { pivotTurn(false);   return; }

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

  if (targetPulses > 0) forward(); else backward();

  unsigned long start = millis();
  unsigned long lastPrintTime = start;   // 用于控制打印频率

  while (avgPulses < needed) {
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

// ========== 电机控制函数 ==========
void forward() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  Serial.println("Forward");
}

void backward() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  Serial.println("Backward");
}

void turnLeft() {
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
  Serial.println("Turn Left");
}

void turnRight() {
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
  Serial.println("Turn Right");
}

void stopAllMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
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

// ==========================================================
// TF：循迹直行（前进），直到到达交叉路口中央
// 使用前端（0,1号）做方向修正，中部（2,3号）判定交叉口
// ==========================================================
void trackForward() {
  unsigned long start = millis();

  while (true) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }

    bool fl = onLine(TRACK_PIN_FL);  // 左前
    bool fr = onLine(TRACK_PIN_FR);  // 右前
    bool ml = onLine(TRACK_PIN_ML);  // 左中
    bool mr = onLine(TRACK_PIN_MR);  // 右中

    // 【交叉口判定】中部左右两个传感器同时压黑：
    // 说明横向黑线已经到达车身中部，车体中心位于路口中央 → 停车
    if (ml && mr) {
      trackFinish("Done");
      return;
    }

    // 【方向修正】理想状态：黑线位于两前端传感器之间，0/1号都在白色区域
    if (fl && fr) {
      // 两前端同时压黑：多为提前碰到路口横线，继续直行交给中部确认
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveForwardSilent();
    } else if (fl) {
      // 仅左前压黑（HIGH）：黑线偏向车体左侧 → 左侧减速向左修正
      setSideSpeed(TRACK_SLOW_SPEED, TRACK_SPEED);
      driveForwardSilent();
    } else if (fr) {
      // 仅右前压黑：黑线偏向车体右侧 → 右侧减速向右修正
      setSideSpeed(TRACK_SPEED, TRACK_SLOW_SPEED);
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
// TB：循迹倒车（后退），直到退至交叉路口中央
// 使用后端（4,5号）做方向修正，中部（2,3号）判定交叉口
// ==========================================================
void trackBackward() {
  unsigned long start = millis();

  while (true) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }

    bool rl = onLine(TRACK_PIN_RL);  // 左后
    bool rr = onLine(TRACK_PIN_RR);  // 右后
    bool ml = onLine(TRACK_PIN_ML);  // 左中
    bool mr = onLine(TRACK_PIN_MR);  // 右中

    // 【交叉口判定】中部左右同时压黑 → 车体中心退到路口中央 → 停车
    if (ml && mr) {
      trackFinish("Done");
      return;
    }

    // 【方向修正】倒车时以后端传感器为准：
    // 后退中车尾向哪边偏，对应侧后端传感器就会压到黑线
    if (rl && rr) {
      // 两后端同时压黑：多为碰到路口横线，继续后退交给中部确认
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
      driveBackwardSilent();
    } else if (rl) {
      // 仅左后压黑：黑线偏向车尾左侧 → 左侧减速，让车尾向左靠回黑线
      setSideSpeed(TRACK_SLOW_SPEED, TRACK_SPEED);
      driveBackwardSilent();
    } else if (rr) {
      // 仅右后压黑：黑线偏向车尾右侧 → 右侧减速修正
      setSideSpeed(TRACK_SPEED, TRACK_SLOW_SPEED);
      driveBackwardSilent();
    } else {
      // 两后端都在白色区域：车尾居中 → 全速后退
      setSideSpeed(TRACK_SPEED, TRACK_SPEED);
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
// PL / PR：交叉路口原地旋转 90 度
// 场景：车体中心已停在标准十字路口中央（通常由 TF/TB 停下）
// 以内侧前端传感器为基准（PL 看左前 0号，PR 看右前 1号）：
//   阶段1：原地旋转，直到该前端传感器离开黑线进入白区（LOW）
//   阶段2：继续旋转，直到该前端传感器再次压到黑线（HIGH），
//           说明车头已扫到新方向的黑线 → 立即停车
// ==========================================================
void pivotTurn(bool leftTurn) {
  unsigned long start = millis();
  setSideSpeed(TRACK_TURN_SPEED, TRACK_TURN_SPEED);

  // 基准前端传感器：左旋看左前（0号），右旋看右前（1号）
  int frontPin = leftTurn ? TRACK_PIN_FL : TRACK_PIN_FR;

  if (leftTurn) rotateLeftSilent();
  else          rotateRightSilent();

  // ---- 阶段1：等待基准前端传感器离开黑线，进入白区（LOW）----
  while (onLine(frontPin)) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  // ---- 阶段2：继续旋转，直到基准前端传感器再次压到黑线（HIGH），
  //             即检测到新方向的黑线 → 停车 ----
  while (!onLine(frontPin)) {
    if (checkEmergencyStop()) { restoreSpeed(); return; }
    if (millis() - start > TRACK_TIMEOUT) { trackFinish("Timeout"); return; }
    delay(2);
  }

  trackFinish("Done");
}