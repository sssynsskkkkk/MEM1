from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from rollout.env.maze import DIRECTION_ORDER, MazeEnv, generate_maze_spec
from rollout.llm_agent.game24_utils import parse_model_output
from rollout.llm_agent.maze_utils import build_prompt, compose_rollout_input, extract_maze_spec


def load_specs(args):
    if args.dataset_parquet:
        dataframe = pd.read_parquet(args.dataset_parquet)
        specs = []
        for _, row in dataframe.head(args.episodes).iterrows():
            specs.append(extract_maze_spec(row["reward_model"]["ground_truth"]))
        return specs

    specs = []
    for offset in range(args.episodes):
        specs.append(
            generate_maze_spec(
                width=args.width,
                height=args.height,
                min_shortest_path=args.min_shortest_path,
                loop_injection_ratio=args.loop_injection_ratio,
                max_steps=args.max_steps,
                max_steps_factor=args.max_steps_factor,
                observation_mode=args.observation_mode,
                reveal_position=args.reveal_position,
                seed=args.seed + offset,
            )
        )
    return specs


def choose_action(env: MazeEnv, policy: str, rng: random.Random) -> str:
    if policy == "oracle":
        remaining_actions = env.shortest_action_sequence()
        if remaining_actions:
            return remaining_actions[0]
        return "UP"
    return rng.choice(DIRECTION_ORDER)


def evaluate_episode(spec, policy: str, rng: random.Random):
    env = MazeEnv(maze_spec=spec)
    observation_block = env.reset()
    base_prompt = build_prompt(spec)
    summary_block = ""
    trace = []

    while not env.done and env.steps_taken < spec["max_steps"]:
        prompt = compose_rollout_input(base_prompt, summary_block, observation_block)
        action = choose_action(env, policy=policy, rng=rng)
        response = f"<summary>step_{env.steps_taken}</summary>\n<action>{action}</action>"
        parsed = parse_model_output(response, require_reasoning=False)
        observation_block, reward, done, info = env.step(parsed.action_text)
        summary_block = parsed.summary_block
        trace.append(
            {
                "prompt": prompt,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        if done:
            break

    ratio = (env.steps_taken / env.optimal_steps) if env.success and env.optimal_steps else None
    return {
        "success": env.success,
        "steps": env.steps_taken,
        "blocked_move_count": sum(1 for row in trace if row["info"]["blocked_move"]),
        "invalid_action_count": sum(1 for row in trace if not row["info"]["action_is_valid"]),
        "unique_cells_visited": len(env.visited),
        "step_ratio": ratio,
        "trace": trace,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight maze MEM1 evaluator and smoke test.")
    parser.add_argument("--dataset-parquet", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", choices=["oracle", "random"], default="oracle")
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--min-shortest-path", type=int, default=10)
    parser.add_argument("--loop-injection-ratio", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-steps-factor", type=float, default=2.0)
    parser.add_argument("--observation-mode", choices=["move_only", "move_plus_local4"], default="move_plus_local4")
    parser.add_argument("--reveal-position", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    specs = load_specs(args)

    results = [evaluate_episode(spec, policy=args.policy, rng=rng) for spec in specs]
    solved = [row for row in results if row["success"]]
    total_steps = sum(row["steps"] for row in results)
    total_blocked = sum(row["blocked_move_count"] for row in results)
    total_invalid = sum(row["invalid_action_count"] for row in results)

    summary = {
        "episodes": len(results),
        "policy": args.policy,
        "success_rate": (len(solved) / len(results)) if results else 0.0,
        "average_steps": (total_steps / len(results)) if results else 0.0,
        "blocked_move_rate": (total_blocked / total_steps) if total_steps else 0.0,
        "invalid_action_rate": (total_invalid / total_steps) if total_steps else 0.0,
        "average_unique_cells_visited": (
            sum(row["unique_cells_visited"] for row in results) / len(results)
        ) if results else 0.0,
        "solved_step_ratio": (
            sum(row["step_ratio"] for row in solved if row["step_ratio"] is not None) / len(solved)
        ) if solved else 0.0,
    }

    print(json.dumps(summary, indent=2))
    if args.show_trace and results:
        print(json.dumps(results[0]["trace"], indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
