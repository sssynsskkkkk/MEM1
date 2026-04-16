from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from rollout.env.maze import MazeEnv, bfs_shortest_path_length, generate_maze_spec, parse_direction_action  # noqa: E402
from rollout.llm_agent.maze_utils import replay_actions  # noqa: E402


def build_fixed_spec():
    return {
        "width": 2,
        "height": 2,
        "start": [0, 0],
        "goal": [1, 1],
        "open_edges": [
            [[0, 0], [0, 1]],
            [[0, 1], [1, 1]],
        ],
        "optimal_path": [[0, 0], [0, 1], [1, 1]],
        "optimal_steps": 2,
        "max_steps": 4,
        "observation_mode": "move_plus_local4",
        "reveal_position": False,
        "step_penalty": -0.01,
        "blocked_move_penalty": -0.02,
        "goal_reward": 1.0,
        "first_visit_bonus": 0.01,
    }


class MazeMem1Tests(unittest.TestCase):
    def test_generated_maze_is_solvable(self):
        spec = generate_maze_spec(width=5, height=5, min_shortest_path=6, seed=7)
        distance = bfs_shortest_path_length(
            tuple(spec["start"]),
            tuple(spec["goal"]),
            spec["height"],
            spec["width"],
            [
                ((left[0], left[1]), (right[0], right[1]))
                for left, right in spec["open_edges"]
            ],
        )
        self.assertIsNotNone(distance)
        self.assertGreaterEqual(distance, 6)
        self.assertEqual(distance, spec["optimal_steps"])

    def test_blocked_and_open_transitions(self):
        env = MazeEnv(maze_spec=build_fixed_spec())
        env.reset()

        observation, reward, done, info = env.step("DOWN")
        self.assertFalse(done)
        self.assertFalse(info["move_succeeded"])
        self.assertTrue(info["blocked_move"])
        self.assertEqual(info["current_position"], (0, 0))
        self.assertAlmostEqual(reward, -0.03, places=6)
        self.assertIn("previous_result: blocked_move", observation)

        observation, reward, done, info = env.step("RIGHT")
        self.assertFalse(done)
        self.assertTrue(info["move_succeeded"])
        self.assertFalse(info["blocked_move"])
        self.assertEqual(info["current_position"], (0, 1))
        self.assertAlmostEqual(reward, 0.0, places=6)
        self.assertIn("local_view:", observation)

    def test_observation_formatting_hides_position_by_default(self):
        env = MazeEnv(maze_spec=build_fixed_spec())
        initial_observation = env.reset()
        self.assertNotIn("current_coordinate:", initial_observation)
        self.assertNotIn("local_view:", initial_observation)

        debug_spec = build_fixed_spec()
        debug_spec["reveal_position"] = True
        debug_env = MazeEnv(maze_spec=debug_spec)
        debug_env.reset()
        observation, _, _, _ = debug_env.step("RIGHT")
        self.assertIn("current_coordinate:", observation)
        self.assertIn("local_view:", observation)

    def test_action_parser(self):
        self.assertEqual(parse_direction_action("  left "), "LEFT")
        self.assertEqual(parse_direction_action("action: up"), "UP")
        with self.assertRaises(ValueError):
            parse_direction_action("jump")

    def test_scripted_successful_episode(self):
        spec = build_fixed_spec()
        env = MazeEnv(maze_spec=spec)
        env.reset()
        actions = env.shortest_action_sequence()
        result = replay_actions(spec, actions)
        self.assertTrue(result.success)
        self.assertEqual(result.total_steps, spec["optimal_steps"])
        self.assertAlmostEqual(result.step_efficiency_ratio, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
