"""
state_machine.py

Robot task state machine.

Workflow:
PATROL
    -> WAIT_CARD
    -> GO_TEACHER
    -> WAIT_LOADING
    -> GO_CHILD
    -> WAIT_UNLOAD
    -> RETURN_PATROL
    -> PATROL
"""

from enum import Enum


class RobotState(Enum):
    """All possible workflow states of the classroom robot."""

    PATROL = 0
    WAIT_CARD = 1
    GO_TEACHER = 2
    WAIT_LOADING = 3
    GO_CHILD = 4
    WAIT_UNLOAD = 5
    RETURN_PATROL = 6


class StateMachine:
    """Manage the current robot state and active delivery task."""

    def __init__(self):
        self.state = RobotState.PATROL
        self.task = None

    def get_state(self):
        """Return the current robot state."""
        return self.state

    def set_state(self, new_state):
        """
        Change the robot state.

        Parameters
        ----------
        new_state : RobotState
            The next state of the robot.
        """

        if not isinstance(new_state, RobotState):
            raise ValueError(
                f"new_state must be RobotState, got {type(new_state)}"
            )

        if self.state == new_state:
            return

        print(f"[STATE] {self.state.name} -> {new_state.name}")

        self.state = new_state

    def is_state(self, state):
        """
        Check whether the robot is currently in a given state.
        """
        return self.state == state

    def set_task(self, task):
        """
        Store the active delivery task.
        """
        self.task = task

        print(f"[STATE] Active task set: {task}")

    def get_task(self):
        """Return the active delivery task."""
        return self.task

    def has_task(self):
        """Return True when an active task exists."""
        return self.task is not None

    def clear_task(self):
        """Remove the active delivery task."""

        if self.task is not None:
            print(f"[STATE] Active task cleared: {self.task}")

        self.task = None

    def reset(self):
        """
        Clear the current task and return to patrol mode.
        """

        self.clear_task()
        self.set_state(RobotState.PATROL)