from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


TRAIN_ROOT = Path(__file__).resolve().parents[2] / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from rollout.env.maze import generate_maze_spec  # noqa: E402
from rollout.llm_agent.maze_utils import build_prompt  # noqa: E402


def build_rows(num_rows: int, split: str, start_index: int, args) -> list[dict]:
    rows = []
    for offset in range(num_rows):
        sample_index = start_index + offset
        spec = generate_maze_spec(
            width=args.width,
            height=args.height,
            min_shortest_path=args.min_shortest_path,
            loop_injection_ratio=args.loop_injection_ratio,
            max_steps=args.max_steps,
            max_steps_factor=args.max_steps_factor,
            observation_mode=args.observation_mode,
            reveal_position=args.reveal_position,
            step_penalty=args.step_penalty,
            blocked_move_penalty=args.blocked_move_penalty,
            goal_reward=args.goal_reward,
            first_visit_bonus=args.first_visit_bonus,
            seed=args.seed + sample_index,
            max_generation_attempts=args.max_generation_attempts,
        )
        rows.append(
            {
                "data_source": "maze",
                "prompt": [{"role": "user", "content": build_prompt(spec)}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": spec,
                },
                "extra_info": {
                    "split": split,
                    "index": sample_index,
                    "maze_seed": spec["maze_seed"],
                    "optimal_steps": spec["optimal_steps"],
                },
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=512)
    parser.add_argument("--num_test", type=int, default=128)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--min_shortest_path", type=int, default=10)
    parser.add_argument("--loop_injection_ratio", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_steps_factor", type=float, default=2.0)
    parser.add_argument("--observation_mode", choices=["move_only", "move_plus_local4"], default="move_plus_local4")
    parser.add_argument("--reveal_position", action="store_true")
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--blocked_move_penalty", type=float, default=-0.02)
    parser.add_argument("--goal_reward", type=float, default=1.0)
    parser.add_argument("--first_visit_bonus", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_generation_attempts", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_rows = build_rows(args.num_train, split="train", start_index=0, args=args)
    test_rows = build_rows(args.num_test, split="test", start_index=len(train_rows), args=args)

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)

    print(
        f"Saved maze MEM1 parquet files to {args.out_dir} "
        f"(train={len(train_rows)}, test={len(test_rows)}, size={args.height}x{args.width})"
    )


if __name__ == "__main__":
    main()
