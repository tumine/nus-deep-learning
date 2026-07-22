"""
state_machine.py

Classroom assistant robot workflow state machine.

Workflow:
PATROL -> 
SCAN -> 
APPROACH_STUDENT -> 
WAIT_CARD -> 
GO_TEACHER -> 
WAIT_LOADING -> 
RETURN_STUDENT -> 
WAIT_UNLOAD -> 
RETURN_PATROL -> 
PATROL
"""

from enum import Enum, auto
from typing import Any, Dict, Optional


class RobotState(Enum):
    PATROL = auto()
    SCAN = auto()
    APPROACH_STUDENT = auto()
    WAIT_CARD = auto()
    GO_TEACHER = auto()
    WAIT_LOADING = auto()
    RETURN_STUDENT = auto()
    WAIT_UNLOAD = auto()
    RETURN_PATROL = auto()


class StateMachine:
    def __init__(self) -> None:
        self.state = RobotState.PATROL
        self.task: Optional[Dict[str, Any]] = None
        self.context: Dict[str, Any] = {
            "route_node": None,
            "scan_direction": None,
            "student_target": None,
            "student_confidence": None,
            "approach_command": None,
        }

    def get_state(self) -> RobotState:
        return self.state

    def is_state(self, state: RobotState) -> bool:
        return self.state == state

    def set_state(self, new_state: RobotState) -> None:
        if not isinstance(new_state, RobotState):
            raise ValueError(f"new_state must be RobotState, got {type(new_state)}")
        if self.state == new_state:
            return
        print(f"[STATE] {self.state.name} -> {new_state.name}")
        self.state = new_state

    def set_task(self, task: Dict[str, Any]) -> None:
        self.task = task
        print(f"[STATE] Active task set: {task}")

    def get_task(self) -> Optional[Dict[str, Any]]:
        return self.task

    def has_task(self) -> bool:
        return self.task is not None

    def clear_task(self) -> None:
        if self.task is not None:
            print(f"[STATE] Active task cleared: {self.task}")
        self.task = None

    def update_context(self, **kwargs: Any) -> None:
        unknown_keys = set(kwargs) - set(self.context)
        if unknown_keys:
            raise KeyError(f"Unknown context key(s): {sorted(unknown_keys)}")
        self.context.update(kwargs)
        for key, value in kwargs.items():
            print(f"[CONTEXT] {key} = {value}")

    def get_context(self) -> Dict[str, Any]:
        return dict(self.context)

    def get_context_value(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def clear_context(self) -> None:
        for key in self.context:
            self.context[key] = None
        print("[CONTEXT] Mission context cleared.")

    def reset(self) -> None:
        self.clear_task()
        self.clear_context()
        self.set_state(RobotState.PATROL)

    def print_status(self) -> None:
        print("=" * 50)
        print(f"State : {self.state.name}")
        print(f"Task  : {self.task}")
        print("Context:")
        for key, value in self.context.items():
            print(f"  {key}: {value}")
        print("=" * 50)