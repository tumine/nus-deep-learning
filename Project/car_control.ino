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