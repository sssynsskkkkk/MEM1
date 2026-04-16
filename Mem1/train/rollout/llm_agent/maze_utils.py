from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from rollout.env.maze import MazeEnv, bfs_shortest_path_length, parse_direction_action
from rollout.env.maze.formatting import format_coordinate
from rollout.env.maze.generator import deserialize_open_edges

from .game24_utils import extract_action_texts


@dataclass
class MazeReplayResult:
    success: bool
    invalid_action_count: int
    blocked_move_count: int
    total_steps: int
    unique_cells_visited: int
    optimal_steps: int
    step_efficiency_ratio: Optional[float]


def extract_maze_spec(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = ["width", "height", "start", "goal", "open_edges"]
    missing = [field for field in required_fields if field not in ground_truth]
    if missing:
        raise ValueError(f"maze ground_truth is missing required fields: {missing}")

    spec = dict(ground_truth)
    spec["width"] = int(spec["width"])
    spec["height"] = int(spec["height"])
    spec["start"] = [int(spec["start"][0]), int(spec["start"][1])]
    spec["goal"] = [int(spec["goal"][0]), int(spec["goal"][1])]
    spec["open_edges"] = [
        [[int(left[0]), int(left[1])], [int(right[0]), int(right[1])]]
        for left, right in spec["open_edges"]
    ]
    spec["max_steps"] = int(spec.get("max_steps", max(spec["width"] * spec["height"], 1)))
    spec["observation_mode"] = spec.get("observation_mode", "move_plus_local4")
    spec["reveal_position"] = bool(spec.get("reveal_position", False))
    spec["step_penalty"] = float(spec.get("step_penalty", -0.01))
    spec["blocked_move_penalty"] = float(spec.get("blocked_move_penalty", -0.02))
    spec["goal_reward"] = float(spec.get("goal_reward", 1.0))
    spec["first_visit_bonus"] = float(spec.get("first_visit_bonus", 0.01))

    if "optimal_steps" not in spec:
        optimal_steps = bfs_shortest_path_length(
            tuple(spec["start"]),
            tuple(spec["goal"]),
            spec["height"],
            spec["width"],
            deserialize_open_edges(spec["open_edges"]),
        )
        if optimal_steps is None:
            raise ValueError("maze ground_truth must describe a solvable maze")
        spec["optimal_steps"] = int(optimal_steps)
    else:
        spec["optimal_steps"] = int(spec["optimal_steps"])

    return spec


def build_prompt(problem: Dict[str, Any]) -> str:
    spec = extract_maze_spec(problem)
    return (
        "You are navigating one partially observable 2D maze with persistent textual working memory.\n"
        "Only the previous <summary> content persists between turns. Use it as compact latent memory.\n"
        "The maze map is hidden. You know only the start coordinate and the goal coordinate.\n"
        "At each turn you may attempt exactly one move.\n\n"
        "At every turn output exactly this shape, with no extra text before or after it:\n"
        "<summary>...</summary>\n"
        "<action>UP|DOWN|LEFT|RIGHT</action>\n\n"
        "Rules:\n"
        "1. The first tag in your response must be <summary>.\n"
        "2. Never skip the <summary> tag. If you have no useful memory yet, write <summary>none</summary>.\n"
        "3. After </summary>, immediately write <action>.\n"
        "4. <action> must be exactly one move: UP, DOWN, LEFT, or RIGHT.\n"
        "5. The environment executes only the action text inside <action>.\n"
        "6. Unless the observation explicitly says current_coordinate, do not assume your exact current coordinate is revealed.\n"
        "7. Use <summary> to keep compact useful memory such as discovered walls, open corridors, dead ends, and route hypotheses.\n"
        "8. The environment state is the source of truth.\n"
        "9. Do not write any text outside the required tags.\n\n"
        f"Start coordinate: {format_coordinate(tuple(spec['start']))}\n"
        f"Goal coordinate: {format_coordinate(tuple(spec['goal']))}\n"
        f"Max steps: {spec['max_steps']}\n"
        f"Observation mode: {spec['observation_mode']}\n"
    )


def compose_rollout_input(base_prompt: str, summary_block: str, observation_block: str) -> str:
    pieces = [base_prompt.strip()]
    if summary_block:
        pieces.append(summary_block.strip())
    if observation_block:
        pieces.append(observation_block.strip())
    return "\n\n".join(piece for piece in pieces if piece)


def replay_actions(ground_truth: Dict[str, Any], actions: Sequence[str], max_total_turns: Optional[int] = None) -> MazeReplayResult:
    spec = extract_maze_spec(ground_truth)
    env = MazeEnv(maze_spec=spec)
    env.reset()

    invalid_action_count = 0
    blocked_move_count = 0

    for action in actions:
        if max_total_turns is not None and env.steps_taken >= max_total_turns:
            break

        _, _, done, info = env.step(action)
        if not info["action_is_valid"]:
            invalid_action_count += 1
        if info["blocked_move"]:
            blocked_move_count += 1
        if done:
            break

    ratio = (env.steps_taken / env.optimal_steps) if env.success and env.optimal_steps else None
    return MazeReplayResult(
        success=env.success,
        invalid_action_count=invalid_action_count,
        blocked_move_count=blocked_move_count,
        total_steps=env.steps_taken,
        unique_cells_visited=len(env.visited),
        optimal_steps=env.optimal_steps,
        step_efficiency_ratio=ratio,
    )


def replay_solution(solution_str: str, ground_truth: Dict[str, Any], max_total_turns: Optional[int] = None) -> MazeReplayResult:
    actions = extract_action_texts(solution_str)
    return replay_actions(ground_truth=ground_truth, actions=actions, max_total_turns=max_total_turns)


__all__ = [
    "MazeReplayResult",
    "build_prompt",
    "compose_rollout_input",
    "extract_maze_spec",
    "parse_direction_action",
    "replay_actions",
    "replay_solution",
]
