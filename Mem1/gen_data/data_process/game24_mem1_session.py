import argparse
import json
import os
import re

import pandas as pd


def normalize_numbers(numbers):
    if isinstance(numbers, list):
        values = [int(x) for x in numbers]
    elif isinstance(numbers, str):
        parts = [part for part in re.split(r"[,\s]+", numbers.strip()) if part]
        values = [int(x) for x in parts]
    else:
        raise TypeError(f"Unsupported numbers type: {type(numbers)}")

    if len(values) != 4:
        raise ValueError(f"Game24 expects exactly 4 numbers, got {values}")
    return values


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["numbers"] = normalize_numbers(item["numbers"])
            item["target"] = int(item.get("target", 24))
            item["max_steps"] = int(item.get("max_steps", 3))
            rows.append(item)
    return rows


def chunk_examples(examples, problems_per_session):
    sessions = []
    for start in range(0, len(examples), problems_per_session):
        session = examples[start : start + problems_per_session]
        if session:
            sessions.append(session)
    return sessions


def build_prompt(first_problem, session_len):
    number_text = ", ".join(str(x) for x in first_problem["numbers"])
    return (
        f"You will solve {session_len} Game-of-24 puzzles in one continuous session.\n"
        "Only the text inside the previous <summary> tag persists across turns and across puzzles. "
        "Treat <summary> as compact working memory, not as an explanation for humans.\n\n"
        "At every turn output exactly this structure and nothing else:\n"
        "<reasoning>...</reasoning>\n"
        "<summary>...</summary>\n"
        "<action>a op b</action>\n\n"
        "Rules:\n"
        "1. <action> must contain exactly one binary operation using two currently available numbers.\n"
        "2. Allowed operators are +, -, *, /.\n"
        "3. Fractions like 8/3 are allowed. Avoid decimals.\n"
        "4. The environment state is the source of truth.\n"
        "5. Use <summary> to store compact durable memory, including failure patterns, bad moves, and useful heuristics.\n"
        "6. If one puzzle fails, keep useful failure notes in <summary> so later puzzles can benefit.\n"
        "7. Do not reset your memory between puzzles inside the same session.\n\n"
        f"Session puzzle 1/{session_len}\n"
        f"Initial numbers: {number_text}\n"
        f"Target: {first_problem.get('target', 24)}\n"
    )


def build_rows(sessions, split, start_index=0):
    rows = []
    for offset, problems in enumerate(sessions):
        session_index = start_index + offset
        rows.append(
            {
                "data_source": "game24",
                "prompt": [{"role": "user", "content": build_prompt(problems[0], len(problems))}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "problems": problems,
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": session_index,
                    "num_problems": len(problems),
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--problems_per_session", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_examples = load_jsonl(args.train_jsonl)
    test_examples = load_jsonl(args.test_jsonl)

    train_sessions = chunk_examples(train_examples, args.problems_per_session)
    test_sessions = chunk_examples(test_examples, args.problems_per_session)

    train_rows = build_rows(train_sessions, split="train", start_index=0)
    test_rows = build_rows(test_sessions, split="test", start_index=len(train_rows))

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)

    print(
        f"Saved sessionized Game24 parquet files to {args.out_dir} "
        f"with problems_per_session={args.problems_per_session}"
    )


if __name__ == "__main__":
    main()
