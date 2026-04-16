from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MazeEnvConfig:
    width: int = 7
    height: int = 7
    min_shortest_path: int = 10
    loop_injection_ratio: float = 0.0
    max_steps: int = 64
    max_steps_factor: float = 2.0
    observation_mode: str = field(
        default="move_plus_local4",
        metadata={"choices": ["move_only", "move_plus_local4"]},
    )
    reveal_position: bool = False
    step_penalty: float = -0.01
    blocked_move_penalty: float = -0.02
    goal_reward: float = 1.0
    first_visit_bonus: float = 0.01
    max_generation_attempts: int = 128
