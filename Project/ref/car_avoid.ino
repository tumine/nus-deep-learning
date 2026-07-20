#include <AFMotor.h>

// ---------- 电机对象 ----------
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// ---------- 超声波引脚 ----------
const int TRIGGER = 22;
const int ECHO = 24;

// ---------- 运动参数 ----------
int speedValue = 230;                // 行驶速度（0~255）
const int OBSTACLE_DIST = 1;        // 障碍物距离阈值（cm）
const int TURN_DURATION = 600;       // 每次左转持续时间（ms）
const int MAX_TURNS = 5;             // 最大连续左转次数（安全保护，仅自动模式）
const unsigned long TIMEOUT = 30000; // 运行超时（30秒，安全保护，仅自动模式）

// ---------- 状态变量 ----------
bool autoMode = false;               // true=自动避障模式，false=手动模式
int turnCount = 0;                  // 连续左转计数（自动模式）
unsigned long startTime;            // 自动模式启动时间
unsigned long lastActionTime = 0;   // 用于非阻塞延时
bool isTurning = false;             // 是否正在执行左转动作（自动模式）

// ---------- 手动避障状态 ----------
enum ManualState {
  MANUAL_IDLE,          // 无手动避障动作
  MANUAL_FORWARD,       // 正在前进且启用避障
  MANUAL_TURNING_LEFT   // 正在执行左转避让
};
ManualState manualState = MANUAL_IDLE;    // 当前手动避障状态
unsigned long manualTurnStartTime = 0;    // 手动左转开始时间

// ---------- 串口命令存储 ----------
String command = ""; // 接收到的完整命令（一行）

void setup() {
  Serial.begin(9600);
  Serial.println("Car Ready! Commands (send each on new line):");
  Serial.println("  F/B/L/R/S  - 手动控制（手动模式下）");
  Serial.println("  A          - 切换 手动/自动 模式");
  Serial.println("In manual mode, 'F' enables obstacle avoidance while moving forward.");

  // 设置电机速度
  motor1.setSpeed(speedValue);
  motor2.setSpeed(speedValue);
  motor3.setSpeed(speedValue);
  motor4.setSpeed(speedValue);
  stopAllMotors();

  // 超声波引脚
  pinMode(TRIGGER, OUTPUT);
  pinMode(ECHO, INPUT);
}

void loop() {
  // ---------- 1. 处理串口命令（任何时候都响应）----------
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n'); // 读取一行命令（直到换行）
    command.trim(); // 去除首尾空白字符（如回车）

    if (command.length() == 0) return; // 忽略空行

    // 处理切换模式命令 'A'
    if (command == "A") {
      autoMode = !autoMode;
      if (autoMode) {
        Serial.println("Switched to AUTO mode (obstacle avoidance)");
        // 重置自动模式相关状态
        turnCount = 0;
        startTime = millis();
        isTurning = false;
        stopAllMotors();
        manualState = MANUAL_IDLE; // 退出手动避障状态
      } else {
        Serial.println("Switched to MANUAL mode");
        stopAllMotors();
        manualState = MANUAL_IDLE;
      }
      return; // 处理完切换命令后跳过本次循环
    }

    // 只有在手动模式下才执行 F/B/L/R/S 命令
    if (!autoMode) {
      if (command == "F") {
        forward();
      } else if (command == "B") {
        backward();
      } else if (command == "L") {
        turnLeft();
      } else if (command == "R") {
        turnRight();
      } else if (command == "S") {
        stopAllMotors();
      } else {
        Serial.println("Unknown command. Use F/B/L/R/S/A");
      }
    } else {
      // 自动模式下忽略其他命令（除了'A'已处理）
      Serial.println("AUTO mode active, ignore manual command. Send A to exit.");
    }
    return; // 处理完命令后，本轮不再执行避障
  }

  // ---------- 2. 如果没有串口命令，根据模式执行避障逻辑 ----------
  if (autoMode) {
    runObstacleAvoidance(); // 自动避障模式
  } else {
    runManualAvoidance();   // 手动模式下的前进避障
  }
}

// ========== 自动避障函数（非阻塞）==========
void runObstacleAvoidance() {
  // 如果正在执行左转动作，则检查是否到达持续时间
  if (isTurning) {
    if (millis() - lastActionTime >= TURN_DURATION) {
      stopAllMotors();
      isTurning = false;
      lastActionTime = millis();
      turnCount++;
    } else {
      return; // 左转尚未完成，继续等待
    }
  }

  // 刚停完车，短停稳定
  if (!isTurning && (millis() - lastActionTime < 200)) {
    return;
  }

  // 测距
  float cm = measureDistance();
  Serial.print("Auto distance: ");
  Serial.print(cm);
  Serial.println(" cm");

  // 决策
  if (cm > 0 && cm < OBSTACLE_DIST) {
    turnLeft();
    isTurning = true;
    lastActionTime = millis();
  } else {
    forward();
    turnCount = 0;
    isTurning = false;
  }

  // 安全停止检查
  if (turnCount >= MAX_TURNS || (millis() - startTime > TIMEOUT)) {
    stopAllMotors();
    Serial.println("Safety stop activated. Switch to MANUAL or reset.");
    autoMode = false;
    isTurning = false;
  }
}

// ========== 手动避障函数（非阻塞）==========
void runManualAvoidance() {
  switch (manualState) {
    case MANUAL_FORWARD: {
      float cm = measureDistance();
      Serial.print("Manual distance: ");
      Serial.print(cm);
      Serial.println(" cm");

      if (cm > 0 && cm < OBSTACLE_DIST) {
        turnLeft();                     // 执行左转
        manualState = MANUAL_TURNING_LEFT;
        manualTurnStartTime = millis();
        Serial.println("Manual: obstacle detected, turning left");
      }
      break;
    }

    case MANUAL_TURNING_LEFT: {
      if (millis() - manualTurnStartTime >= TURN_DURATION) {
        forward();                     // 恢复前进
        manualState = MANUAL_FORWARD;
        Serial.println("Manual: turn finished, resume forward");
      }
      break;
    }

    default:
      // MANUAL_IDLE：不做任何事
      break;
  }
}

// ========== 测距通用函数 ==========
float measureDistance() {
  digitalWrite(TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER, LOW);
  unsigned long duration = pulseIn(ECHO, HIGH, 30000);
  //if (duration == 0) {
  //  return -1; // 无回波视为无障碍
  //}
  return duration / 58.2;
}

// ========== 电机控制函数（同时管理手动状态）==========
void forward() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  Serial.println("Forward");
  if (!autoMode) {
    manualState = MANUAL_FORWARD;   // 进入前进避障状态
  }
}

void backward() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  Serial.println("Backward");
  if (!autoMode) {
    manualState = MANUAL_IDLE;
  }
}

void turnLeft() {
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
  Serial.println("Turn Left");
  if (!autoMode) {
    manualState = MANUAL_IDLE;
  }
}

void turnRight() {
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
  Serial.println("Turn Right");
  if (!autoMode) {
    manualState = MANUAL_IDLE;
  }
}

void stopAllMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
  Serial.println("Stopped");
  if (!autoMode) {
    manualState = MANUAL_IDLE;
  }
}