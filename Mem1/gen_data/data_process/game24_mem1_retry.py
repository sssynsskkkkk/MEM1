import argparse
import json
import os
import random
import re

import pandas as pd


DEFAULT_NUMBER_FIELDS = ["numbers", "cards", "nums", "digits", "values"]
DEFAULT_TEXT_FIELDS = ["question", "puzzle", "input", "prompt"]


def normalize_numbers(numbers):
    if isinstance(numbers, (list, tuple)):
        values = [int(x) for x in numbers]
    elif isinstance(numbers, str):
        parts = [part for part in re.split(r"[,\s]+", numbers.strip()) if part]
        values = [int(x) for x in parts]
    else:
        raise TypeError(f"Unsupported numbers type: {type(numbers)}")

    if len(values) != 4:
        raise ValueError(f"Game24 expects exactly 4 numbers, got {values}")
    return values


def infer_numbers(example, numbers_field=None):
    candidate_fields = []
    if numbers_field:
        candidate_fields.append(numbers_field)
    candidate_fields.extend(field for field in DEFAULT_NUMBER_FIELDS if field not in candidate_fields)

    for field in candidate_fields:
        if field in example and example[field] is not None:
            return normalize_numbers(example[field])

    for field in DEFAULT_TEXT_FIELDS:
        value = example.get(field)
        if not isinstance(value, str):
            continue
        matches = re.findall(r"-?\d+", value)
        if len(matches) == 4:
            return [int(token) for token in matches]

    raise ValueError(
        f"Could not infer four Game24 numbers from example keys: {sorted(example.keys())}. "
        "Use --numbers_field if the dataset uses a custom field."
    )


def normalize_problem(example, numbers_field=None, target_field=None, max_steps=3):
    target_value = 24
    if target_field and target_field in example and example[target_field] is not None:
        target_value = int(example[target_field])
    elif "target" in example and example["target"] is not None:
        target_value = int(example["target"])

    local_max_steps = int(example.get("max_steps", max_steps))
    return {
        "numbers": infer_numbers(example, numbers_field=numbers_field),
        "target": target_value,
        "max_steps": local_max_steps,
    }


def load_jsonl(path, numbers_field=None, target_field=None, max_steps=3):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                normalize_problem(
                    item,
                    numbers_field=numbers_field,
                    target_field=target_field,
                    max_steps=max_steps,
                )
            )
    return rows


def split_examples(examples, test_size, seed):
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples to create train/test split")

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_count = max(1, int(round(len(examples) * test_size)))
    test_count = min(test_count, len(examples) - 1)
    test_indices = set(indices[:test_count])

    train_examples = [examples[idx] for idx in range(len(examples)) if idx not in test_indices]
    test_examples = [examples[idx] for idx in range(len(examples)) if idx in test_indices]
    return train_examples, test_examples


def load_hf_splits(dataset_name, dataset_config=None, split_name=None, test_size=0.1, seed=42, numbers_field=None, target_field=None, max_steps=3):
    import datasets

    dataset = datasets.load_dataset(dataset_name, dataset_config)

    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset and "test" in dataset:
            train_examples = [
                normalize_problem(row, numbers_field=numbers_field, target_field=target_field, max_steps=max_steps)
                for row in dataset["train"]
            ]
            test_examples = [
                normalize_problem(row, numbers_field=numbers_field, target_field=target_field, max_steps=max_steps)
                for row in dataset["test"]
            ]
            return train_examples, test_examples

        if split_name:
            if split_name not in dataset:
                raise ValueError(f"Requested split '{split_name}' not found. Available splits: {list(dataset.keys())}")
            source_split = dataset[split_name]
        elif len(dataset) == 1:
            source_split = next(iter(dataset.values()))
        elif "train" in dataset:
            source_split = dataset["train"]
        else:
            raise ValueError(f"Could not choose a source split automatically from {list(dataset.keys())}")
    else:
        source_split = dataset

    examples = [
        normalize_problem(row, numbers_field=numbers_field, target_field=target_field, max_steps=max_steps)
        for row in source_split
    ]
    return split_examples(examples, test_size=test_size, seed=seed)


def build_prompt(problem):
    number_text = ", ".join(str(x) for x in problem["numbers"])
    return (
        "You are solving one Game-of-24 puzzle with persistent textual working memory.\n"
        "Only the previous <summary> content persists between turns. Use it as compact latent memory.\n"
        "If an attempt fails, the environment resets to the original four numbers, but your <summary> remains and should help the next attempt.\n\n"
        "At every turn output exactly this shape, with no extra text before or after it:\n"
        "<summary>...</summary>\n"
        "<action>a op b</action>\n\n"
        "Rules:\n"
        "1. The first tag in your response must be <summary>.\n"
        "2. Never skip the <summary> tag. If you have no useful memory yet, write <summary>none</summary>.\n"
        "3. After </summary>, immediately write <action>.\n"
        "4. <action> must be exactly one binary operation using two currently available numbers.\n"
        "5. The environment executes only the action text inside <action>.\n"
        "6. Allowed operators: +, -, *, /\n"
        "7. Fractions are allowed. Avoid decimals.\n"
        "8. The environment state is the source of truth.\n"
        "9. Use <summary> to keep compact useful memory such as failed branches, invalid moves, and promising constructions.\n"
        "10. If the attempt ends with one number that is not 24, keep the lesson in <summary> and try the same puzzle again.\n"
        "11. Do not write any text outside the required tags.\n\n"
        f"Initial numbers: {number_text}\n"
        f"Target: {problem.get('target', 24)}\n"
    )


def build_rows(examples, split, start_index=0):
    rows = []
    for offset, problem in enumerate(examples):
        sample_index = start_index + offset
        rows.append(
            {
                "data_source": "game24",
                "prompt": [{"role": "user", "content": build_prompt(problem)}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "numbers": problem["numbers"],
                        "target": problem["target"],
                        "max_steps": problem["max_steps"],
                    },
                },
                "extra_info": {
                    "split": split,
                    "index": sample_index,
                },
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str)
    parser.add_argument("--test_jsonl", type=str)
    parser.add_argument("--hf_dataset", type=str)
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default=None)
    parser.add_argument("--numbers_field", type=str, default=None)
    parser.add_argument("--target_field", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=3)
    return parser.parse_args()


def load_examples(args):
    if args.hf_dataset:
        return load_hf_splits(
            dataset_name=args.hf_dataset,
            dataset_config=args.hf_config,
            split_name=args.hf_split,
            test_size=args.test_size,
            seed=args.seed,
            numbers_field=args.numbers_field,
            target_field=args.target_field,
            max_steps=args.max_steps,
        )

    if args.train_jsonl and args.test_jsonl:
        train_examples = load_jsonl(
            args.train_jsonl,
            numbers_field=args.numbers_field,
            target_field=args.target_field,
            max_steps=args.max_steps,
        )
        test_examples = load_jsonl(
            args.test_jsonl,
            numbers_field=args.numbers_field,
            target_field=args.target_field,
            max_steps=args.max_steps,
        )
        return train_examples, test_examples

    raise ValueError("Provide either --hf_dataset or both --train_jsonl and --test_jsonl")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_examples, test_examples = load_examples(args)
    train_rows = build_rows(train_examples, split="train", start_index=0)
    test_rows = build_rows(test_examples, split="test", start_index=len(train_rows))

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)

    print(
        f"Saved single-puzzle retry Game24 parquet files to {args.out_dir} "
        f"(train={len(train_rows)}, test={len(test_rows)})"
    )


if __name__ == "__main__":
    main()
