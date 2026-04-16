from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..base import BaseLanguageBasedEnv
from .actions import DIRECTION_ORDER, move_coordinate, parse_direction_action
from .config import MazeEnvConfig
from .formatting import format_observation
from .generator import (
    bfs_shortest_path,
    build_adjacency,
    deserialize_open_edges,
    generate_maze_spec,
    in_bounds,
    path_to_actions,
)


Coord = Tuple[int, int]


def _to_coord(values) -> Coord:
    return int(values[0]), int(values[1])


class MazeEnv(BaseLanguageBasedEnv):
    def __init__(self, config: Optional[MazeEnvConfig] = None, maze_spec: Optional[Dict] = None) -> None:
        self.config = config or MazeEnvConfig()
        self._base_maze_spec = maze_spec

        self.maze_spec: Dict = {}
        self.height = 0
        self.width = 0
        self.start: Coord = (0, 0)
        self.goal: Coord = (0, 0)
        self.position: Coord = (0, 0)
        self.adjacency: Dict[Coord, set[Coord]] = {}
        self.optimal_steps = 0
        self.max_steps = 0
        self.observation_mode = self.config.observation_mode
        self.reveal_position = self.config.reveal_position
        self.step_penalty = self.config.step_penalty
        self.blocked_move_penalty = self.config.blocked_move_penalty
        self.goal_reward = self.config.goal_reward
        self.first_visit_bonus = self.config.first_visit_bonus
        self.steps_taken = 0
        self.total_reward = 0.0
        self.done = False
        self.success = False
        self.visited: set[Coord] = set()
        self.last_observation = ""
        self.last_result = "episode_start"

    def _resolve_maze_spec(self, seed=None, maze_spec: Optional[Dict] = None) -> Dict:
        spec = maze_spec or self._base_maze_spec
        if spec is None:
            spec = generate_maze_spec(
                width=self.config.width,
                height=self.config.height,
                min_shortest_path=self.config.min_shortest_path,
                loop_injection_ratio=self.config.loop_injection_ratio,
                max_steps=self.config.max_steps,
                max_steps_factor=self.config.max_steps_factor,
                observation_mode=self.config.observation_mode,
                reveal_position=self.config.reveal_position,
                step_penalty=self.config.step_penalty,
                blocked_move_penalty=self.config.blocked_move_penalty,
                goal_reward=self.config.goal_reward,
                first_visit_bonus=self.config.first_visit_bonus,
                seed=seed,
                max_generation_attempts=self.config.max_generation_attempts,
            )
        return dict(spec)

    def reset(self, seed=None, maze_spec: Optional[Dict] = None, **kwargs):
        del kwargs
        self.maze_spec = self._resolve_maze_spec(seed=seed, maze_spec=maze_spec)
        self.height = int(self.maze_spec["height"])
        self.width = int(self.maze_spec["width"])
        self.start = _to_coord(self.maze_spec["start"])
        self.goal = _to_coord(self.maze_spec["goal"])
        self.position = self.start
        self.adjacency = build_adjacency(self.height, self.width, deserialize_open_edges(self.maze_spec["open_edges"]))
        self.optimal_steps = int(
            self.maze_spec.get("optimal_steps")
            or max(len(bfs_shortest_path(self.start, self.goal, self.height, self.width, deserialize_open_edges(self.maze_spec["open_edges"]))) - 1, 1)
        )
        self.max_steps = int(self.maze_spec.get("max_steps", self.config.max_steps))
        self.observation_mode = self.maze_spec.get("observation_mode", self.config.observation_mode)
        self.reveal_position = bool(self.maze_spec.get("reveal_position", self.config.reveal_position))
        self.step_penalty = float(self.maze_spec.get("step_penalty", self.config.step_penalty))
        self.blocked_move_penalty = float(self.maze_spec.get("blocked_move_penalty", self.config.blocked_move_penalty))
        self.goal_reward = float(self.maze_spec.get("goal_reward", self.config.goal_reward))
        self.first_visit_bonus = float(self.maze_spec.get("first_visit_bonus", self.config.first_visit_bonus))

        self.steps_taken = 0
        self.total_reward = 0.0
        self.done = False
        self.success = False
        self.visited = {self.position}
        self.last_result = "episode_start"
        self.last_observation = format_observation(
            turn_index=0,
            turns_left=self.max_steps,
            previous_result=self.last_result,
            goal_reached=False,
            observation_mode=self.observation_mode,
            reveal_position=self.reveal_position,
            position=self.position,
        )
        return self.last_observation

    def get_local_view(self, coord: Optional[Coord] = None) -> Dict[str, bool]:
        position = coord or self.position
        local_view: Dict[str, bool] = {}
        for direction in DIRECTION_ORDER:
            neighbor = move_coordinate(position, direction)
            local_view[direction] = in_bounds(neighbor, self.height, self.width) and neighbor in self.adjacency[position]
        return local_view

    def shortest_path(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> List[Coord]:
        return bfs_shortest_path(
            start or self.position,
            goal or self.goal,
            self.height,
            self.width,
            deserialize_open_edges(self.maze_spec["open_edges"]),
        )

    def shortest_action_sequence(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> List[str]:
        path = self.shortest_path(start=start, goal=goal)
        return path_to_actions(path)

    def step(self, action: str):
        if self.done:
            info = {
                "action_is_valid": False,
                "move_succeeded": False,
                "blocked_move": False,
                "visited_new_cell": False,
                "success": self.success,
                "goal_reached": self.success,
                "turn_index": self.steps_taken,
                "turns_left": max(self.max_steps - self.steps_taken, 0),
                "current_position": self.position,
                "unique_cells_visited": len(self.visited),
                "optimal_steps": self.optimal_steps,
                "path_efficiency_ratio": (self.steps_taken / self.optimal_steps) if self.success and self.optimal_steps else None,
            }
            return self.last_observation, 0.0, True, info

        reward = self.step_penalty
        action_is_valid = True
        move_succeeded = False
        blocked_move = False
        visited_new_cell = False

        action_label = (action or "").strip()
        try:
            direction = parse_direction_action(action_label)
            action_label = direction
        except ValueError:
            direction = None
            action_is_valid = False

        if direction is None:
            previous_result = "invalid_action"
        else:
            candidate = move_coordinate(self.position, direction)
            if not in_bounds(candidate, self.height, self.width) or candidate not in self.adjacency[self.position]:
                blocked_move = True
                reward += self.blocked_move_penalty
                previous_result = "blocked_move"
            else:
                self.position = candidate
                move_succeeded = True
                previous_result = "move_succeeded"
                if self.position not in self.visited:
                    visited_new_cell = True
                    self.visited.add(self.position)
                    reward += self.first_visit_bonus
                if self.position == self.goal:
                    reward += self.goal_reward
                    self.success = True
                    previous_result = "goal_reached"

        self.steps_taken += 1
        if self.success:
            self.done = True
        elif self.steps_taken >= self.max_steps:
            self.done = True
            previous_result = "step_limit_reached"

        self.last_result = previous_result
        turns_left = max(self.max_steps - self.steps_taken, 0)
        local_view = self.get_local_view() if self.observation_mode == "move_plus_local4" else None
        self.last_observation = format_observation(
            turn_index=self.steps_taken,
            turns_left=turns_left,
            previous_result=previous_result,
            last_action=action_label or "INVALID",
            goal_reached=self.success,
            observation_mode=self.observation_mode,
            local_view=local_view,
            reveal_position=self.reveal_position,
            position=self.position,
        )
        self.total_reward += reward

        info = {
            "action_is_valid": action_is_valid,
            "move_succeeded": move_succeeded,
            "blocked_move": blocked_move,
            "visited_new_cell": visited_new_cell,
            "success": self.success,
            "goal_reached": self.success,
            "turn_index": self.steps_taken,
            "turns_left": turns_left,
            "current_position": self.position,
            "unique_cells_visited": len(self.visited),
            "optimal_steps": self.optimal_steps,
            "path_efficiency_ratio": (self.steps_taken / self.optimal_steps) if self.success and self.optimal_steps else None,
        }
        return self.last_observation, reward, self.done, info

    def render(self, mode: str = "text"):
        del mode
        return self.last_observation

    def close(self):
        return None
