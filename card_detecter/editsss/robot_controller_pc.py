"""
robot_controller.py

Robot action controller.
"""

class RobotController:

    def __init__(self):

        self.busy = False

    def execute(self, command):

        self.busy = True

        print(f"[Robot] Executing {command}")

        # TODO
        # send tcp

        # TODO
        # wait finish

        self.busy = False