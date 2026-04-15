import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from datasets import DownloadConfig, load_dataset

from rollout.llm_agent.game24_utils import apply_action_to_state, format_fraction, format_state, parse_model_output


DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

LEADING_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

PREVIOUS_RESULT_KEYS = [
    "continue",
    "solved",
    "retry_unparsable_output",
    "retry_invalid_action",
    "retry_step_limit",
    "retry_wrong_final_value",
]


def build_model_slug(model_id):
    base = model_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def build_prompt(problem, prepend_no_think=True):
    number_text = ", ".join(str(x) for x in problem["numbers"])
    prompt = (
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
    if prepend_no_think:
        return f"/no_think\n{prompt}"
    return prompt


def build_generation_text(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            pass
    return prompt


def build_state_block(state, target, steps_used, max_steps, attempt_index, previous_result):
    turns_left = max(max_steps - steps_used, 0)
    lines = [
        "<observation>",
        f"attempt_index: {attempt_index + 1}",
        f"remaining_numbers: [{format_state(state)}]",
        f"target: {format_fraction(target)}",
        f"turns_left_in_current_attempt: {turns_left}",
    ]
    if previous_result:
        lines.append(f"previous_result: {previous_result}")
    lines.append("</observation>")
    return "\n".join(lines)


def compose_rollout_input(base_prompt, summary_block, observation_block):
    pieces = [base_prompt.strip()]
    if summary_block:
        pieces.append(summary_block.strip())
    if observation_block:
        pieces.append(observation_block.strip())
    return "\n\n".join(piece for piece in pieces if piece)


def normalize_previous_result_label(previous_result):
    if previous_result.startswith("retry_invalid_action:"):
        return "retry_invalid_action"
    if previous_result.startswith("retry_wrong_final_value:"):
        return "retry_wrong_final_value"
    return previous_result


def load_problems(dataset_name, split, cache_dir, local_files_only, max_samples, seed):
    download_config = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        download_config=download_config,
    )
    numbers_field = "numbers" if "numbers" in dataset.column_names else "nums"
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen = sorted(indices[: min(max_samples, len(indices))])

    problems = []
    for idx in chosen:
        row = dataset[idx]
        problems.append(
            {
                "sample_index": idx,
                "numbers": row[numbers_field],
                "target": 24,
            }
        )
    return problems


def strip_leading_think_block(text):
    return LEADING_THINK_RE.sub("", text, count=1)


def analyze_response(raw_response, numbers, require_reasoning, parser_mode):
    raw_text = raw_response.strip()
    stripped_think_text = strip_leading_think_block(raw_text).strip()
    selected_text = stripped_think_text if parser_mode == "strip_think" else raw_text

    raw_parsed = parse_model_output(raw_text, require_reasoning=require_reasoning)
    selected_parsed = parse_model_output(selected_text, require_reasoning=require_reasoning)
    state = [Fraction(str(value)) for value in numbers]
    action_outcome = (
        apply_action_to_state(state, selected_parsed.action_text, tolerance=1e-5)
        if selected_parsed.action_text
        else None
    )

    return {
        "raw_text": raw_text,
        "stripped_think_text": stripped_think_text,
        "selected_text": selected_text,
        "starts_with_think": raw_text.startswith("<think>"),
        "has_summary_tag": "<summary>" in raw_text,
        "has_action_tag": "<action>" in raw_text,
        "raw_parser_accept": raw_parsed.format_ok,
        "parser_accept": selected_parsed.format_ok,
        "think_strip_helped": (not raw_parsed.format_ok) and selected_parsed.format_ok,
        "summary_present": bool(selected_parsed.summary_block),
        "action_present": bool(selected_parsed.action_block),
        "summary_text": selected_parsed.summary_text,
        "action_text": selected_parsed.action_text,
        "action_valid": action_outcome.is_valid if action_outcome else False,
        "action_error": action_outcome.error if action_outcome else "",
        "overall_compliance": selected_parsed.format_ok and bool(action_outcome and action_outcome.is_valid),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_response(model, tokenizer, text, max_new_tokens):
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    completion_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()


def load_model_and_tokenizer(model_id, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()
    return model, tokenizer, torch


def cleanup_model(model, torch_module):
    del model
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def evaluate_format_model(model_id, problems, args):
    print(f"\n### Train-parser format eval: {model_id} ###")
    model, tokenizer, torch_module = load_model_and_tokenizer(model_id, args)

    counts = {
        "starts_with_think": 0,
        "has_summary_tag": 0,
        "has_action_tag": 0,
        "raw_parser_accept": 0,
        "parser_accept": 0,
        "think_strip_helped": 0,
        "summary_present": 0,
        "action_present": 0,
        "action_valid": 0,
        "overall_compliance": 0,
    }
    traces = []
    started_at = time.time()

    for sample_idx, problem in enumerate(problems, start=1):
        prompt = build_prompt(problem, prepend_no_think=args.prepend_no_think)
        text = build_generation_text(tokenizer, prompt)
        response = generate_response(model, tokenizer, text, args.format_max_new_tokens)
        metrics = analyze_response(response, problem["numbers"], args.require_reasoning, args.parser_mode)

        for key in counts:
            if metrics[key]:
                counts[key] += 1

        traces.append(
            {
                "sample_index": problem["sample_index"],
                "model_id": model_id,
                "numbers": problem["numbers"],
                "prompt": prompt,
                "response": response,
                **metrics,
            }
        )

        if sample_idx % args.log_every == 0 or sample_idx == len(problems):
            accept_rate = (counts["parser_accept"] / sample_idx) * 100 if sample_idx else 0.0
            print(
                f"[{model_id}] format {sample_idx}/{len(problems)} | "
                f"{args.parser_mode} parser accept: {accept_rate:.2f}%"
            )

    duration_sec = time.time() - started_at
    total = len(problems)
    trace_path = Path(args.output_dir) / "trainparser_format_traces" / f"{build_model_slug(model_id)}_format.jsonl"
    write_jsonl(trace_path, traces)
    cleanup_model(model, torch_module)

    return {
        "model_id": model_id,
        "starts_with_think_rate": counts["starts_with_think"] / total if total else 0.0,
        "has_summary_tag_rate": counts["has_summary_tag"] / total if total else 0.0,
        "has_action_tag_rate": counts["has_action_tag"] / total if total else 0.0,
        "raw_parser_accept_rate": counts["raw_parser_accept"] / total if total else 0.0,
        "parser_accept_rate": counts["parser_accept"] / total if total else 0.0,
        "think_strip_helped_rate": counts["think_strip_helped"] / total if total else 0.0,
        "summary_present_rate": counts["summary_present"] / total if total else 0.0,
        "action_present_rate": counts["action_present"] / total if total else 0.0,
        "action_valid_rate": counts["action_valid"] / total if total else 0.0,
        "overall_compliance_rate": counts["overall_compliance"] / total if total else 0.0,
        "evaluated_samples": total,
        "duration_sec": duration_sec,
        "trace_jsonl_path": str(trace_path),
    }


def evaluate_env_model(model_id, problems, args):
    print(f"\n### Train-parser env eval: {model_id} ###")
    model, tokenizer, torch_module = load_model_and_tokenizer(model_id, args)

    outcome_counts = {key: 0 for key in PREVIOUS_RESULT_KEYS}
    generated_turns = 0
    raw_parser_accept_turns = 0
    parser_accept_turns = 0
    valid_action_turns = 0
    retry_counts_total = 0
    solved_samples = 0
    traces = []
    started_at = time.time()

    for sample_idx, problem in enumerate(problems, start=1):
        base_prompt = build_prompt(problem, prepend_no_think=args.prepend_no_think)
        initial_state = [Fraction(str(value)) for value in problem["numbers"]]
        current_state = list(initial_state)
        target = Fraction(str(problem["target"]))
        carried_summary_block = ""
        retry_count = 0
        steps_in_attempt = 0
        previous_result = "attempt_start"
        sample_solved = False

        for turn_idx in range(1, args.env_max_turns + 1):
            observation_block = build_state_block(
                current_state,
                target,
                steps_in_attempt,
                max(len(initial_state) - 1, 1),
                retry_count,
                previous_result,
            )
            prompt = compose_rollout_input(base_prompt, carried_summary_block, observation_block)
            text = build_generation_text(tokenizer, prompt)
            response = generate_response(model, tokenizer, text, args.env_max_new_tokens)
            metrics = analyze_response(response, current_state, args.require_reasoning, args.parser_mode)

            generated_turns += 1
            if metrics["raw_parser_accept"]:
                raw_parser_accept_turns += 1
            if metrics["parser_accept"]:
                parser_accept_turns += 1

            turn_outcome = "retry_unparsable_output"
            done = False

            if metrics["parser_accept"]:
                if metrics["action_valid"]:
                    valid_action_turns += 1
                    carried_summary_block = f"<summary>{metrics['summary_text']}</summary>"
                    action_outcome = apply_action_to_state(current_state, metrics["action_text"], tolerance=1e-5)
                    current_state = action_outcome.next_state
                    steps_in_attempt += 1

                    if len(current_state) == 1:
                        if abs(float(current_state[0] - target)) <= 1e-5:
                            turn_outcome = "solved"
                            sample_solved = True
                            done = True
                        else:
                            turn_outcome = f"retry_wrong_final_value:{format_fraction(current_state[0])}"
                            retry_count += 1
                            current_state = list(initial_state)
                            steps_in_attempt = 0
                    elif steps_in_attempt >= max(len(initial_state) - 1, 1):
                        turn_outcome = "retry_step_limit"
                        retry_count += 1
                        current_state = list(initial_state)
                        steps_in_attempt = 0
                    else:
                        turn_outcome = "continue"
                else:
                    if metrics["summary_text"]:
                        carried_summary_block = f"<summary>{metrics['summary_text']}</summary>"
                    turn_outcome = f"retry_invalid_action:{metrics['action_error']}"
                    retry_count += 1
                    current_state = list(initial_state)
                    steps_in_attempt = 0
            else:
                turn_outcome = "retry_unparsable_output"
                retry_count += 1
                current_state = list(initial_state)
                steps_in_attempt = 0

            normalized_outcome = normalize_previous_result_label(turn_outcome)
            if normalized_outcome in outcome_counts:
                outcome_counts[normalized_outcome] += 1

            traces.append(
                {
                    "sample_index": problem["sample_index"],
                    "turn_index": turn_idx,
                    "model_id": model_id,
                    "numbers": problem["numbers"],
                    "prompt": prompt,
                    "response": response,
                    **metrics,
                    "previous_result_after_turn": turn_outcome,
                    "retry_count_after_turn": retry_count,
                    "state_after_turn": [format_fraction(v) for v in current_state],
                    "done": done,
                }
            )

            previous_result = turn_outcome
            if done:
                break

        retry_counts_total += retry_count
        if sample_solved:
            solved_samples += 1

        if sample_idx % args.log_every == 0 or sample_idx == len(problems):
            accept_rate = (parser_accept_turns / generated_turns) * 100 if generated_turns else 0.0
            print(
                f"[{model_id}] env {sample_idx}/{len(problems)} | "
                f"{args.parser_mode} parser accept turns: {accept_rate:.2f}%"
            )

    duration_sec = time.time() - started_at
    total = len(problems)
    trace_path = Path(args.output_dir) / "trainparser_env_traces" / f"{build_model_slug(model_id)}_env.jsonl"
    write_jsonl(trace_path, traces)
    cleanup_model(model, torch_module)

    return {
        "model_id": model_id,
        "raw_parser_accept_turn_rate": raw_parser_accept_turns / generated_turns if generated_turns else 0.0,
        "parser_accept_turn_rate": parser_accept_turns / generated_turns if generated_turns else 0.0,
        "valid_action_turn_rate": valid_action_turns / generated_turns if generated_turns else 0.0,
        "success_rate": solved_samples / total if total else 0.0,
        "avg_generated_turns": generated_turns / total if total else 0.0,
        "avg_retry_count": retry_counts_total / total if total else 0.0,
        "continue_rate": outcome_counts["continue"] / generated_turns if generated_turns else 0.0,
        "solved_turn_rate": outcome_counts["solved"] / generated_turns if generated_turns else 0.0,
        "retry_unparsable_output_rate": outcome_counts["retry_unparsable_output"] / generated_turns if generated_turns else 0.0,
        "retry_invalid_action_rate": outcome_counts["retry_invalid_action"] / generated_turns if generated_turns else 0.0,
        "retry_step_limit_rate": outcome_counts["retry_step_limit"] / generated_turns if generated_turns else 0.0,
        "retry_wrong_final_value_rate": outcome_counts["retry_wrong_final_value"] / generated_turns if generated_turns else 0.0,
        "evaluated_samples": total,
        "duration_sec": duration_sec,
        "trace_jsonl_path": str(trace_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Game24 format/env behavior using the exact MEM1 training parser.")
    parser.add_argument("--dataset", default="nlile/24-game")
    parser.add_argument("--split", default="train")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", default="/workspace/MEM1/Mem1/train/eval_dumps/game24-mem1-trainparser-eval")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--shared-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format-max-new-tokens", type=int, default=256)
    parser.add_argument("--env-max-new-tokens", type=int, default=160)
    parser.add_argument("--env-max-turns", type=int, default=3)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepend-no-think", dest="prepend_no_think", action="store_true")
    parser.add_argument("--no-prepend-no-think", dest="prepend_no_think", action="store_false")
    parser.add_argument("--require-reasoning", action="store_true")
    parser.add_argument("--parser-mode", choices=["raw", "strip_think"], default="raw")
    parser.set_defaults(prepend_no_think=False, require_reasoning=False)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(
        args.dataset,
        args.split,
        args.cache_dir,
        args.local_files_only,
        args.shared_samples,
        args.seed,
    )

    print(f"Saving outputs to: {output_dir}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Shared samples: {len(problems)}")
    print(f"prepend_no_think: {args.prepend_no_think}")
    print(f"require_reasoning: {args.require_reasoning}")
    print(f"parser_mode: {args.parser_mode}")

    format_results = []
    env_results = []
    for model_id in args.models:
        format_results.append(evaluate_format_model(model_id, problems, args))
        env_results.append(evaluate_env_model(model_id, problems, args))

    format_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "shared_samples": len(problems),
        "seed": args.seed,
        "prepend_no_think": args.prepend_no_think,
        "require_reasoning": args.require_reasoning,
        "parser_mode": args.parser_mode,
        "results": format_results,
    }
    env_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "shared_samples": len(problems),
        "seed": args.seed,
        "prepend_no_think": args.prepend_no_think,
        "require_reasoning": args.require_reasoning,
        "parser_mode": args.parser_mode,
        "results": env_results,
    }

    format_json_path = output_dir / "trainparser_format_summary.json"
    format_csv_path = output_dir / "trainparser_format_summary.csv"
    env_json_path = output_dir / "trainparser_env_summary.json"
    env_csv_path = output_dir / "trainparser_env_summary.csv"
    write_json(format_json_path, format_payload)
    write_csv(format_csv_path, format_results)
    write_json(env_json_path, env_payload)
    write_csv(env_csv_path, env_results)

    print("\n--- Train-Parser Format Summary ---")
    for row in format_results:
        print(
            f"Model: {row['model_id']} | "
            f"starts_with_think={row['starts_with_think_rate'] * 100:.1f}% | "
            f"summary_tag={row['has_summary_tag_rate'] * 100:.1f}% | "
            f"action_tag={row['has_action_tag_rate'] * 100:.1f}% | "
            f"raw_parser_accept={row['raw_parser_accept_rate'] * 100:.1f}% | "
            f"parser_accept={row['parser_accept_rate'] * 100:.1f}% | "
            f"think_strip_helped={row['think_strip_helped_rate'] * 100:.1f}% | "
            f"action_valid={row['action_valid_rate'] * 100:.1f}%"
        )

    print("\n--- Train-Parser Environment Summary ---")
    for row in env_results:
        print(
            f"Model: {row['model_id']} | "
            f"raw_parser_accept_turn={row['raw_parser_accept_turn_rate'] * 100:.1f}% | "
            f"parser_accept_turn={row['parser_accept_turn_rate'] * 100:.1f}% | "
            f"retry_unparsable={row['retry_unparsable_output_rate'] * 100:.1f}% | "
            f"retry_invalid={row['retry_invalid_action_rate'] * 100:.1f}% | "
            f"valid_action_turn={row['valid_action_turn_rate'] * 100:.1f}% | "
            f"success={row['success_rate'] * 100:.1f}%"
        )

    print(f"\nSaved format JSON: {format_json_path}")
    print(f"Saved format CSV : {format_csv_path}")
    print(f"Saved env JSON   : {env_json_path}")
    print(f"Saved env CSV    : {env_csv_path}")


if __name__ == "__main__":
    main()
