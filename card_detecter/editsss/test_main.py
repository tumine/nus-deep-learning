from robot_controller import RobotController

robot = RobotController()

task = {
    "type": "GO_TO_TEACHER",
    "item": "blocks",
    "student_position": (320, 180)
}

robot.execute