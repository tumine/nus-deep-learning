#include <AFMotor.h>
#include <Wire.h>
#include <PID_v1.h>

// ---------- 电机对象 ----------
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// ---------- 超声波引脚 ----------
const int TRIGGER = 22;
const int ECHO = 24;

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

// ---------- PID 参数（直线纠偏）----------
float pidKp = 1.2;
float pidKi = 0.02;
float pidKd = 0.1;
float pidError = 0;
float pidIntegral = 0;
float pidPrevError = 0;

// ---------- MPU-6500 传感器（I2C）----------
const int MPU6500_ADDR = 0x68;        // MPU-6500 I2C 地址（AD0 接 GND）
float accelX, accelY, accelZ;         // 加速度（g）
float gyroX, gyroY, gyroZ;            // 角速度（deg/s）
float gyroZBias = 0;                  // Z 轴陀螺仪零偏

// 陀螺仪量程 ±1000 deg/s（灵敏度 32.8 LSB/(deg/s)）
// 全速原地旋转时角速度可能超过 ±250 deg/s，量程过小会饱和，导致积分角度严重偏小
const float GYRO_SENS = 32.8;
unsigned long i2cErrorCount = 0;      // I2C 通讯失败计数（诊断用）
unsigned int mpuResetCount = 0;       // 检测到 MPU-6500 复位的次数（诊断用）

// ---------- 转向 PID 参数 ----------
float turnKp = 2.95;                   // 比例系数
float turnKi = 0.3;                   // 积分系数
float turnKd = 0.8;                   // 微分系数
const int TURN_BASE_SPEED = 180;      // 转向基础速度，PID 在此基础上加减

// ---------- 串口命令存储 ----------
String command = "";

void setup() {
  Serial.begin(9600);
  Serial.println("Car Ready! Commands:");
  Serial.println("  F/B/L/R/S - Manual control");
  Serial.println("  O - Drive forward until obstacle, then send 'D'");
  Serial.println("  G [dist] - Move distance (cm), positive forward, negative backward");
  Serial.println("  T [angle] - Turn angle (degrees), positive left, negative right");
  Serial.println("  (After G/T completes, 'Done' is sent)");
  Serial.println("  RST - Reset encoder counts");
  Serial.println("  M - Gyro debug (no motors, rotate by hand)");

  // 初始化 I2C 和 MPU-6500 陀螺仪
  Wire.begin();
  Wire.setWireTimeout(3000, true);  // 防止 I2C 卡死（若旧版内核编译报错可删除本行）
  initMPU6500();

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
}

void loop() {
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
    else if (cmd == 'M') {
      debugGyro();
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
void encoderLeftISR() {
  encoderCountLeft++;
}
void encoderRightISR() {
  encoderCountRight++;
}

// ========== 直线移动（距离 cm）+ PID 航向修正 ==========
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

  // 确定方向
  bool isForward = (targetPulses > 0);
  if (isForward) {
    motor1.run(FORWARD);
    motor2.run(FORWARD);
    motor3.run(FORWARD);
    motor4.run(FORWARD);
  } else {
    motor1.run(BACKWARD);
    motor2.run(BACKWARD);
    motor3.run(BACKWARD);
    motor4.run(BACKWARD);
  }

  unsigned long start = millis();
  unsigned long lastPrintTime = start;

  // PID 变量清零
  pidIntegral = 0;
  pidPrevError = 0;

  while (avgPulses < needed) {
    noInterrupts();
    long left = abs(encoderCountLeft);
    long right = abs(encoderCountRight);
    interrupts();
    avgPulses = (left + right) / 2;

    // -------- PID 计算修正量 --------
    pidError = left - right;  // 左轮脉冲数 - 右轮脉冲数（正=左偏快，需减速左轮）
    pidIntegral += pidError;
    float derivative = pidError - pidPrevError;
    pidPrevError = pidError;
    float correction = pidKp * pidError + pidKi * pidIntegral + pidKd * derivative;

    // 限幅，防止速度突变
    if (correction > 60) correction = 60;
    if (correction < -60) correction = -60;

    // 计算左右轮实际速度
    int leftSpeed = speedValue - correction;
    int rightSpeed = speedValue + correction;

    // 限幅到 0~255
    if (leftSpeed < 0) leftSpeed = 0;
    if (leftSpeed > 255) leftSpeed = 255;
    if (rightSpeed < 0) rightSpeed = 0;
    if (rightSpeed > 255) rightSpeed = 255;

    // 设置四个电机速度（注意：AFMotor的setSpeed作用于对应电机）
    motor1.setSpeed(rightSpeed);   // 右前
    motor2.setSpeed(rightSpeed);   // 右后
    motor3.setSpeed(leftSpeed);    // 左后
    motor4.setSpeed(leftSpeed);    // 左前

    // ---------- 原有调试打印（保持不变） ----------
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

  // 最终打印（保持不变）
  Serial.print("Final L:");
  Serial.print(abs(encoderCountLeft));
  Serial.print(" R:");
  Serial.print(abs(encoderCountRight));
  Serial.print(" Avg:");
  Serial.println((abs(encoderCountLeft) + abs(encoderCountRight)) / 2);

  Serial.println("Done");
}

// ========== 基于 MPU-6500 和 PID 的精确闭环原地旋转 ==========
// angle_deg > 0 → 左转（逆时针）
// angle_deg < 0 → 右转（顺时针）
void turnAngle(float angle_deg) {
  float targetAngle = abs(angle_deg);
  if (targetAngle < 0.5) {
    Serial.println("Angle too small");
    return;
  }

  double turnInput = 0;
  double turnOutput = 0;
  double turnSetpoint = targetAngle;

  PID turnPID(&turnInput, &turnOutput, &turnSetpoint, turnKp, turnKi, turnKd, DIRECT);
  turnPID.SetMode(AUTOMATIC);
  
  // 【修复1】允许PID接管全部速度范围（-255 到 255）
  // 正输出代表需要继续转，负输出代表超调需要反转修正
  turnPID.SetOutputLimits(-255, 255);
  turnPID.SetSampleTime(10); 

  // 记录预期的基础转向（左转还是右转）
  bool isLeftTurn = (angle_deg > 0);

  float yaw = 0;                      
  unsigned long lastTime = micros();  
  unsigned long start = millis();     
  unsigned long lastPrintTime = start;
  int settleCount = 0;
  const int SETTLE_NEEDED = 8;  

  while (true) {
    float gz = readGyroZOnly();

    // 【修复】NaN 保护：I2C 连续失败时跳过本次积分，保持上次 yaw 不变
    if (isnan(gz)) {
      // 检测 MPU-6500 是否因电机全速导致电源跌落而进入 SLEEP 模式
      ensureMPU6500Awake();
      delay(5);
      continue;
    }

    unsigned long now = micros();
    float dt = (now - lastTime) / 1000000.0;
    lastTime = now;
    if (dt > 0.05) dt = 0.01;

    // 【修复2】方向感知积分：如果正在往预期方向转，角度增加；如果正在反转修正，角度减少
    if (turnOutput >= 0) {
      yaw += abs(gz) * dt;
    } else {
      yaw -= abs(gz) * dt;
    }

    turnInput = yaw;

    if (turnPID.Compute()) {
      // 【修复3】直接使用 PID 输出的绝对值作为电机速度，移除 TURN_BASE_SPEED
      int motorSpeed = constrain(abs((int)turnOutput), 0, 255);
      
      // 添加死区补偿：如果速度太小电机转不动，给一个启动底速（根据你的车体重量调整）
      if (motorSpeed > 0 && motorSpeed < 130) {
          motorSpeed = 130; 
      }

      // 【修复4】动态方向控制：如果 turnOutput < 0，说明越过了目标，需要临时反转方向
      bool currentTurnLeft = isLeftTurn;
      if (turnOutput < 0) {
        currentTurnLeft = !currentTurnLeft; 
      }

      if (currentTurnLeft) {
        // 左转动作
        motor1.run(BACKWARD); motor2.run(FORWARD);
        motor3.run(FORWARD); motor4.run(BACKWARD);
      } else {
        // 右转动作
        motor1.run(FORWARD); motor2.run(BACKWARD);
        motor3.run(BACKWARD); motor4.run(FORWARD);
      }

      motor1.setSpeed(motorSpeed);
      motor2.setSpeed(motorSpeed);
      motor3.setSpeed(motorSpeed);
      motor4.setSpeed(motorSpeed);
    }

    if (millis() - lastPrintTime >= 100) {
      Serial.print("Yaw:"); Serial.print(yaw);
      Serial.print(" Target:"); Serial.print(targetAngle);
      Serial.print(" Out:"); Serial.print(turnOutput);
      Serial.print(" Gyro:"); Serial.println(abs(gz));
      lastPrintTime = millis();
    }

    float angleError = abs(targetAngle - yaw);
    if (angleError < 1.0 && abs(gz) < 3.0) {
      settleCount++;
      if (settleCount >= SETTLE_NEEDED) break;
    } else {
      settleCount = 0;  
    }

    if (millis() - start > 30000) {
      Serial.println("Turn timeout");
      break;
    }
    delay(1);
  }

  stopAllMotors();
  Serial.print("Final Yaw: ");
  Serial.println(yaw);
  Serial.println("Done");
}

// ========== 唤醒并配置 MPU-6500（可重复调用，用于复位后自动恢复）==========
void wakeMPU6500() {
  // 步骤 1：唤醒 MPU-6500（退出睡眠模式）
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 寄存器地址
  Wire.write(0x00);  // 写入 0x00，唤醒传感器
  Wire.endTransmission(true);

  // 步骤 2：配置陀螺仪量程为 ±1000 deg/s（灵敏度 32.8 LSB/(deg/s)）
  // 全速原地旋转角速度可超过 ±250 deg/s，量程不足会饱和，导致积分角度严重偏小
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG 寄存器地址
  Wire.write(0x10);  // 0x10 = ±1000 deg/s
  Wire.endTransmission(true);

  // 步骤 3：配置加速度计量程为 ±2g（灵敏度 16384 LSB/g，默认值）
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG 寄存器地址
  Wire.write(0x00);  // 0x00 = ±2g（默认值）
  Wire.endTransmission(true);

  // 步骤 4：开启 DLPF（陀螺仪带宽 44Hz），抑制电机振动引入的高频噪声
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x1A);  // CONFIG 寄存器地址
  Wire.write(0x03);  // DLPF_CFG = 3
  Wire.endTransmission(true);
}

// ========== MPU-6500 初始化（含陀螺仪零偏校准）==========
void initMPU6500() {
  wakeMPU6500();

  delay(100);  // 等待传感器稳定

  // 陀螺仪零偏校准
  // 小车静止状态下采样 200 次，取平均值作为零偏
  long sumGyroZ = 0;
  for (int i = 0; i < 200; i++) {
    readMPU6500Raw();
    sumGyroZ += gyroZ;
    delay(5);
  }
  gyroZBias = (float)sumGyroZ / 200.0;

  Serial.print("MPU-6500 OK, GyroZ bias: ");
  Serial.println(gyroZBias);
}

// ========== 轻量级陀螺仪 Z 轴读取（专用于转向 PID 循环）==========
// 只读取 GYRO_ZOUT_H (0x47) 和 GYRO_ZOUT_L (0x48)，共 2 字节
// 相比 readMPU6500()（14 字节），I2C 耗时减少约 85%
//
// 【方案 A】使用 Repeated START 模式（endTransmission(false)）替代分离写/读事务：
//   - STOP 条件可能导致 MPU-6500 内部寄存器指针复位，导致读到错误寄存器
//   - Repeated START 在读期间保持总线占用，确保寄存器指针不会因 STOP 而复位
//   - 配合 3 次重试，失败时返回 NaN，由调用方跳过本次积分（避免 0 值污染）
//   - readMPU6500() 中 14 字节 Repeated START 失败，但 2 字节的 readGyroZOnly() 更稳定
float readGyroZOnly() {
  for (uint8_t attempt = 0; attempt < 3; attempt++) {
    // 步骤 1：写入寄存器地址，使用 false（Repeated START），保持总线占用
    Wire.beginTransmission(MPU6500_ADDR);
    Wire.write(0x47);                     // GYRO_ZOUT_H
    if (Wire.endTransmission(false) != 0) {  // false = 不发送 STOP，保持 Repeated START
      i2cErrorCount++;
      continue;                         // I2C 通讯错误，重试
    }

    // 延时让 MPU-6500 准备数据（datasheet 要求 ≥1.25μs，3μs 留余量）
    delayMicroseconds(3);

    // 步骤 2：在同一事务中读取 2 字节，最后发送 STOP
    Wire.requestFrom(MPU6500_ADDR, 2, true);
    if (Wire.available() < 2) {
      i2cErrorCount++;
      continue;                         // 数据不足（可能传感器离线），重试
    }

    uint8_t gzH = Wire.read();
    uint8_t gzL = Wire.read();

    // 合成 16 位有符号整数并转为 deg/s，减去零偏消除漂移
    int16_t raw = (int16_t)((gzH << 8) | gzL);
    return (float)raw / GYRO_SENS - gyroZBias;
  }
  return NAN;                           // 连续失败：返回 NaN
}

// ========== MPU-6500 看门狗：检测复位/SLEEP 并自动恢复 ==========
// 电机全速时若电源跌落，MPU-6500 会复位并回到 SLEEP 模式（PWR_MGMT_1 = 0x40），
// 此后陀螺仪输出寄存器恒为 0，读到的"角速度"恒等于 -零偏（表现为固定值，例如 6.09）
void ensureMPU6500Awake() {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  if (Wire.endTransmission(true) != 0) {
    i2cErrorCount++;
    return;
  }
  Wire.requestFrom(MPU6500_ADDR, 1, true);
  if (Wire.available() < 1) {
    i2cErrorCount++;
    return;
  }
  uint8_t pwr = Wire.read();
  if (pwr & 0x40) {        // SLEEP 位被置位 → 传感器曾经复位
    mpuResetCount++;
    wakeMPU6500();         // 重新唤醒并恢复配置（零偏沿用，无需重新校准）
  }
}

// ========== 陀螺仪诊断：不开电机，连续打印 10 秒 ==========
// 用法：发送 M，然后用手将小车原地转动约 90°，
// 若 YawInt 能积到约 ±90，说明传感器与积分链路正常，
// 问题出在电机运行时的电源跌落/电磁干扰
void debugGyro() {
  Serial.println("Gyro debug 10s (motors OFF), rotate by hand...");
  float yaw = 0;
  unsigned long lastTime = micros();
  unsigned long start = millis();
  unsigned long lastPrint = start;
  while (millis() - start < 10000) {
    float gz = readGyroZOnly();
    unsigned long now = micros();
    float dt = (now - lastTime) / 1000000.0;
    lastTime = now;
    if (dt > 0.05) dt = 0.01;
    if (!isnan(gz)) yaw += gz * dt;    // 有符号积分，正反方向可互相抵消
    if (millis() - lastPrint >= 100) {
      Serial.print("GyroZ:");
      Serial.print(isnan(gz) ? 999.0 : gz);   // 999 表示本次读取失败
      Serial.print(" YawInt:");
      Serial.print(yaw);
      Serial.print(" Err:");
      Serial.println(i2cErrorCount);
      lastPrint = millis();
    }
    ensureMPU6500Awake();
    delay(5);
  }
  Serial.println("Debug done");
}

// ========== 读取 MPU-6500 原始数据 ==========
// 仅读取原始寄存器值，不做零偏修正（用于校准阶段）
void readMPU6500Raw() {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x43);  // 从 GYRO_XOUT_H 开始读取（跳过加速度和温度）
  Wire.endTransmission(true);   // 【修复】使用 STOP 替代 repeated start，避免寄存器指针设置失败
  Wire.requestFrom(MPU6500_ADDR, 6, true);  // 只读陀螺仪 6 字节

  uint8_t gxH = Wire.read(), gxL = Wire.read();
  uint8_t gyH = Wire.read(), gyL = Wire.read();
  uint8_t gzH = Wire.read(), gzL = Wire.read();

  // 将 2 字节有符号整数转为物理单位 deg/s（灵敏度需与 GYRO_CONFIG 量程一致）
  gyroX = (float)((int16_t)((gxH << 8) | gxL)) / GYRO_SENS;
  gyroY = (float)((int16_t)((gyH << 8) | gyL)) / GYRO_SENS;
  gyroZ = (float)((int16_t)((gzH << 8) | gzL)) / GYRO_SENS;
}

// ========== 读取 MPU-6500 数据（含零偏修正）==========
// 读取全部传感器数据，并对 gyroZ 减去校准零偏
void readMPU6500() {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(0x3B);  // 从 ACCEL_XOUT_H 开始，连续读取 14 字节
  Wire.endTransmission(true);   // 【修复】使用 STOP 替代 repeated start，避免寄存器指针设置失败
  Wire.requestFrom(MPU6500_ADDR, 14, true);

  // --- 加速度计（6 字节，3 轴 × 2 字节）---
  uint8_t axH = Wire.read(), axL = Wire.read();
  uint8_t ayH = Wire.read(), ayL = Wire.read();
  uint8_t azH = Wire.read(), azL = Wire.read();

  // --- 温度（2 字节，跳过）---
  Wire.read(); Wire.read();

  // --- 陀螺仪（6 字节，3 轴 × 2 字节）---
  uint8_t gxH = Wire.read(), gxL = Wire.read();
  uint8_t gyH = Wire.read(), gyL = Wire.read();
  uint8_t gzH = Wire.read(), gzL = Wire.read();

  // 合成 16 位有符号整数，并转换为物理单位
  // 陀螺仪：灵敏度由 GYRO_SENS 定义，需与 GYRO_CONFIG 量程一致
  // 减去零偏消除漂移
  gyroX = (float)((int16_t)((gxH << 8) | gxL)) / GYRO_SENS;
  gyroY = (float)((int16_t)((gyH << 8) | gyL)) / GYRO_SENS;
  gyroZ = (float)((int16_t)((gzH << 8) | gzL)) / GYRO_SENS - gyroZBias;

  // 加速度计：±2g 量程 → 16384.0 LSB/g（暂时保留以备后续扩展）
  accelX = (float)((int16_t)((axH << 8) | axL)) / 16384.0;
  accelY = (float)((int16_t)((ayH << 8) | ayL)) / 16384.0;
  accelZ = (float)((int16_t)((azH << 8) | azL)) / 16384.0;
}

// ========== 直行直到障碍物 ==========
void driveUntilObstacle() {
  // 1. 开始探测前，先清零编码器
  noInterrupts();
  encoderCountLeft = 0;
  encoderCountRight = 0;
  interrupts();

  while (true) {
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

// ========== 电机控制函数（保持不变）==========
void forward() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  Serial.println("Forward");
}

void backward() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  Serial.println("Backward");
}

void turnLeft() {
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  Serial.println("Turn Left");
}

void turnRight() {
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  Serial.println("Turn Right");
}

void stopAllMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
  Serial.println("Stopped");
}
