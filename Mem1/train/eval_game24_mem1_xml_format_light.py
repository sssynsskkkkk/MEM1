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
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

STRICT_SUMMARY_ACTION_RE = re.compile(
    r"^\s*<summary>.*?</summary>\s*<action>.*?</action>\s*$",
    re.DOTALL,
)

PREVIOUS_RESULT_KEYS = [
    "continue",
    "solved",
    "retry_unparsable_output",
    "retry_invalid_action",
    "retry_step_limit",
    "retry_wrong_final_value",
]


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


def build_model_slug(model_id):
    base = model_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


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


def evaluate_response(raw_response, numbers):
    parsed = parse_model_output(raw_response, require_reasoning=False)
    stripped = raw_response.strip()
    env_format_ok = parsed.format_ok
    strict_xml_ok = bool(STRICT_SUMMARY_ACTION_RE.match(stripped))
    summary_first_ok = stripped.startswith("<summary>")
    summary_present = bool(parsed.summary_block)
    action_present = bool(parsed.action_block)
    action_parse_ok = False
    action_valid_on_initial_numbers = False
    action_error = ""

    if parsed.action_text:
        state = [Fraction(str(value)) for value in numbers]
        action_outcome = apply_action_to_state(state, parsed.action_text, tolerance=1e-5)
        action_parse_ok = "invalid action format" not in action_outcome.error
        action_valid_on_initial_numbers = action_outcome.is_valid
        action_error = action_outcome.error

    overall_compliance = (
        strict_xml_ok
        and summary_first_ok
        and summary_present
        and action_present
        and action_valid_on_initial_numbers
    )

    return {
        "env_format_ok": env_format_ok,
        "strict_xml_ok": strict_xml_ok,
        "summary_first_ok": summary_first_ok,
        "summary_present": summary_present,
        "action_present": action_present,
        "action_parse_ok": action_parse_ok,
        "action_valid_on_initial_numbers": action_valid_on_initial_numbers,
        "overall_compliance": overall_compliance,
        "summary_text": parsed.summary_text,
        "action_text": parsed.action_text,
        "action_error": action_error,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "env_format_rate",
                "strict_xml_rate",
                "summary_first_rate",
                "summary_present_rate",
                "action_present_rate",
                "action_parse_rate",
                "action_valid_rate",
                "overall_compliance_rate",
                "success_rate",
                "env_format_valid_turn_rate",
                "valid_action_turn_rate",
                "avg_generated_turns",
                "avg_retry_count",
                "continue_rate",
                "solved_turn_rate",
                "retry_unparsable_output_rate",
                "retry_invalid_action_rate",
                "retry_step_limit_rate",
                "retry_wrong_final_value_rate",
                "evaluated_samples",
                "duration_sec",
                "trace_jsonl_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_results(results, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    labels = [row["model_id"].split("/")[-1] for row in results]
    strict_values = [row["overall_compliance_rate"] * 100 for row in results]
    env_values = [row["env_format_valid_turn_rate"] * 100 for row in results]
    retry_values = [row["retry_unparsable_output_rate"] * 100 for row in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plots = [
        (axes[0], strict_values, "Strict Prompt Compliance (%)", "Strict XML + Valid Action"),
        (axes[1], env_values, "Environment-Accepted Turns (%)", "Env Parsable + State-Valid"),
        (axes[2], retry_values, "Retry Unparsable (%)", "retry_unparsable_output"),
    ]

    for ax, values, ylabel, title in plots:
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylim(0, max(100, max(values, default=0) + 5))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = output_dir / "xml_env_compliance_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def plot_format_results(results, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    labels = [row["model_id"].split("/")[-1] for row in results]
    strict_values = [row["strict_xml_rate"] * 100 for row in results]
    overall_values = [row["overall_compliance_rate"] * 100 for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plots = [
        (axes[0], strict_values, "Strict XML Rate (%)", "Exact <summary><action> Format"),
        (axes[1], overall_values, "Overall Compliance (%)", "Strict XML + State-Valid Action"),
    ]

    for ax, values, ylabel, title in plots:
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylim(0, max(100, max(values, default=0) + 5))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = output_dir / "xml_format_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def write_csv_generic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_init_wandb(args):
    if not args.wandb_project:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config={
            "dataset": args.dataset,
            "split": args.split,
            "models": args.models,
            "shared_samples": args.shared_samples,
            "seed": args.seed,
            "format_max_new_tokens": args.format_max_new_tokens,
            "env_max_new_tokens": args.env_max_new_tokens,
            "env_max_turns": args.env_max_turns,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
        },
    )
    return run


def log_wandb_model_row(run, row, index):
    if run is None:
        return

    import wandb

    log_row = {
        "model_index": index,
        "model_id": row["model_id"],
        "env_format_rate": row["env_format_rate"],
        "strict_xml_rate": row["strict_xml_rate"],
        "summary_first_rate": row["summary_first_rate"],
        "summary_present_rate": row["summary_present_rate"],
        "action_present_rate": row["action_present_rate"],
        "action_parse_rate": row["action_parse_rate"],
        "action_valid_rate": row["action_valid_rate"],
        "overall_compliance_rate": row["overall_compliance_rate"],
        "success_rate": row["success_rate"],
        "env_format_valid_turn_rate": row["env_format_valid_turn_rate"],
        "valid_action_turn_rate": row["valid_action_turn_rate"],
        "avg_generated_turns": row["avg_generated_turns"],
        "avg_retry_count": row["avg_retry_count"],
        "continue_rate": row["continue_rate"],
        "solved_turn_rate": row["solved_turn_rate"],
        "retry_unparsable_output_rate": row["retry_unparsable_output_rate"],
        "retry_invalid_action_rate": row["retry_invalid_action_rate"],
        "retry_step_limit_rate": row["retry_step_limit_rate"],
        "retry_wrong_final_value_rate": row["retry_wrong_final_value_rate"],
        "evaluated_samples": row["evaluated_samples"],
        "duration_sec": row["duration_sec"],
    }
    run.log(log_row)

    model_key = build_model_slug(row["model_id"])
    for key, value in log_row.items():
        if key in {"model_index", "model_id"}:
            continue
        run.summary[f"{model_key}/{key}"] = value


def log_wandb_prefixed_row(run, row, index, prefix):
    if run is None:
        return

    log_row = {"model_index": index, "model_id": row["model_id"]}
    for key, value in row.items():
        if key in {"model_id", "trace_jsonl_path"}:
            continue
        log_row[f"{prefix}/{key}"] = value
    run.log(log_row)

    model_key = build_model_slug(row["model_id"])
    for key, value in row.items():
        if key in {"model_id", "trace_jsonl_path"}:
            continue
        run.summary[f"{prefix}/{model_key}/{key}"] = value


def finalize_wandb(run, payload, results, json_path, csv_path, plot_path):
    if run is None:
        return

    import wandb

    table_columns = [
        "model_id",
        "env_format_rate",
        "strict_xml_rate",
        "summary_first_rate",
        "summary_present_rate",
        "action_present_rate",
        "action_parse_rate",
        "action_valid_rate",
        "overall_compliance_rate",
        "success_rate",
        "env_format_valid_turn_rate",
        "valid_action_turn_rate",
        "avg_generated_turns",
        "avg_retry_count",
        "continue_rate",
        "solved_turn_rate",
        "retry_unparsable_output_rate",
        "retry_invalid_action_rate",
        "retry_step_limit_rate",
        "retry_wrong_final_value_rate",
        "evaluated_samples",
        "duration_sec",
    ]
    table = wandb.Table(columns=table_columns)
    for row in results:
        table.add_data(*[row[col] for col in table_columns])

    run.log({"summary_table": table})
    run.summary["summary_json_path"] = str(json_path)
    run.summary["summary_csv_path"] = str(csv_path)
    if plot_path:
        run.log({"xml_env_compliance_plot": wandb.Image(str(plot_path))})

    wandb.save(str(json_path), base_path=str(json_path.parent))
    wandb.save(str(csv_path), base_path=str(csv_path.parent))
    if plot_path:
        wandb.save(str(plot_path), base_path=str(plot_path.parent))

    run.finish()


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
    print(f"\n### Evaluating XML format only: {model_id} ###")
    model, tokenizer, torch_module = load_model_and_tokenizer(model_id, args)

    counts = {
        "env_format_ok": 0,
        "strict_xml_ok": 0,
        "summary_first_ok": 0,
        "summary_present": 0,
        "action_present": 0,
        "action_parse_ok": 0,
        "action_valid_on_initial_numbers": 0,
        "overall_compliance": 0,
    }
    traces = []
    started_at = time.time()

    for sample_idx, problem in enumerate(problems, start=1):
        prompt = build_prompt(problem)
        text = build_generation_text(tokenizer, prompt)
        response = generate_response(model, tokenizer, text, args.format_max_new_tokens)
        metrics = evaluate_response(response, problem["numbers"])

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
                "env_format_ok": metrics["env_format_ok"],
                "strict_xml_ok": metrics["strict_xml_ok"],
                "summary_first_ok": metrics["summary_first_ok"],
                "summary_present": metrics["summary_present"],
                "action_present": metrics["action_present"],
                "action_parse_ok": metrics["action_parse_ok"],
                "action_valid_on_initial_numbers": metrics["action_valid_on_initial_numbers"],
                "overall_compliance": metrics["overall_compliance"],
                "summary_text": metrics["summary_text"],
                "action_text": metrics["action_text"],
                "action_error": metrics["action_error"],
            }
        )

        if sample_idx % args.log_every == 0 or sample_idx == len(problems):
            strict_rate = (counts["strict_xml_ok"] / sample_idx) * 100 if sample_idx else 0.0
            print(f"[{model_id}] format {sample_idx}/{len(problems)} | strict xml: {strict_rate:.2f}%")

    duration_sec = time.time() - started_at
    total = len(problems)
    trace_path = Path(args.output_dir) / "format_only_traces" / f"{build_model_slug(model_id)}_format_only.jsonl"
    write_jsonl(trace_path, traces)
    cleanup_model(model, torch_module)

    return {
        "model_id": model_id,
        "strict_xml_rate": counts["strict_xml_ok"] / total if total else 0.0,
        "env_format_rate": counts["env_format_ok"] / total if total else 0.0,
        "summary_first_rate": counts["summary_first_ok"] / total if total else 0.0,
        "summary_present_rate": counts["summary_present"] / total if total else 0.0,
        "action_present_rate": counts["action_present"] / total if total else 0.0,
        "action_parse_rate": counts["action_parse_ok"] / total if total else 0.0,
        "action_valid_rate": counts["action_valid_on_initial_numbers"] / total if total else 0.0,
        "overall_compliance_rate": counts["overall_compliance"] / total if total else 0.0,
        "evaluated_samples": total,
        "duration_sec": duration_sec,
        "trace_jsonl_path": str(trace_path),
    }


def evaluate_model(model_id, problems, args):
    print(f"\n### Evaluating environment rollout: {model_id} ###")
    model, tokenizer, torch_module = load_model_and_tokenizer(model_id, args)

    counts = {
        "env_format_ok": 0,
        "strict_xml_ok": 0,
        "summary_first_ok": 0,
        "summary_present": 0,
        "action_present": 0,
        "action_parse_ok": 0,
        "action_valid_on_initial_numbers": 0,
        "overall_compliance": 0,
    }
    outcome_counts = {key: 0 for key in PREVIOUS_RESULT_KEYS}
    generated_turns = 0
    env_format_valid_turns = 0
    valid_action_turns = 0
    retry_counts_total = 0
    solved_samples = 0
    traces = []
    started_at = time.time()

    for sample_idx, problem in enumerate(problems, start=1):
        base_prompt = build_prompt(problem)
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
            metrics = evaluate_response(response, current_state)

            generated_turns += 1
            if turn_idx == 1:
                for key in counts:
                    if metrics[key]:
                        counts[key] += 1

            turn_outcome = "retry_unparsable_output"
            done = False

            if metrics["env_format_ok"]:
                env_format_valid_turns += 1
                if metrics["action_valid_on_initial_numbers"]:
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
                    "env_format_ok": metrics["env_format_ok"],
                    "strict_xml_ok": metrics["strict_xml_ok"],
                    "summary_first_ok": metrics["summary_first_ok"],
                    "summary_present": metrics["summary_present"],
                    "action_present": metrics["action_present"],
                    "action_parse_ok": metrics["action_parse_ok"],
                    "action_valid_on_initial_numbers": metrics["action_valid_on_initial_numbers"],
                    "overall_compliance": metrics["overall_compliance"],
                    "summary_text": metrics["summary_text"],
                    "action_text": metrics["action_text"],
                    "action_error": metrics["action_error"],
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
            running_rate = (counts["overall_compliance"] / sample_idx) * 100 if sample_idx else 0.0
            print(
                f"[{model_id}] {sample_idx}/{len(problems)} samples processed | "
                f"strict compliance: {running_rate:.2f}%"
            )

    duration_sec = time.time() - started_at
    total = len(problems)

    trace_path = Path(args.output_dir) / "xml_format_traces" / f"{build_model_slug(model_id)}_xml_format.jsonl"
    write_jsonl(trace_path, traces)

    cleanup_model(model, torch_module)

    return {
        "model_id": model_id,
        "env_format_rate": counts["env_format_ok"] / total if total else 0.0,
        "strict_xml_rate": counts["strict_xml_ok"] / total if total else 0.0,
        "summary_first_rate": counts["summary_first_ok"] / total if total else 0.0,
        "summary_present_rate": counts["summary_present"] / total if total else 0.0,
        "action_present_rate": counts["action_present"] / total if total else 0.0,
        "action_parse_rate": counts["action_parse_ok"] / total if total else 0.0,
        "action_valid_rate": counts["action_valid_on_initial_numbers"] / total if total else 0.0,
        "overall_compliance_rate": counts["overall_compliance"] / total if total else 0.0,
        "success_rate": solved_samples / total if total else 0.0,
        "env_format_valid_turn_rate": env_format_valid_turns / generated_turns if generated_turns else 0.0,
        "valid_action_turn_rate": valid_action_turns / generated_turns if generated_turns else 0.0,
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
    parser = argparse.ArgumentParser(description="Lightweight MEM1 Game24 XML/environment evaluation.")
    parser.add_argument("--dataset", default="nlile/24-game")
    parser.add_argument("--split", default="train")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", default="/workspace/MEM1/Mem1/train/eval_dumps/game24-mem1-xml-format-light")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--shared-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format-max-new-tokens", type=int, default=192)
    parser.add_argument("--env-max-new-tokens", type=int, default=128)
    parser.add_argument("--env-max-turns", type=int, default=4)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-format", action="store_true")
    parser.add_argument("--skip-env", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = maybe_init_wandb(args)

    print(f"Saving outputs to: {output_dir}")
    print(f"Dataset: {args.dataset} ({args.split})")

    format_results = []
    env_results = []
    format_json_path = None
    format_csv_path = None
    format_plot_path = None
    env_json_path = None
    env_csv_path = None
    env_plot_path = None
    shared_problems = None

    if not args.skip_format or not args.skip_env:
        shared_problems = load_problems(
            args.dataset,
            args.split,
            args.cache_dir,
            args.local_files_only,
            args.shared_samples,
            args.seed,
        )
        print(f"Shared sampled problems: {len(shared_problems)}")

    if not args.skip_format:
        print(f"Format-only evaluation: {len(shared_problems)} sampled problems")
        for model_index, model_id in enumerate(args.models):
            row = evaluate_format_model(model_id, shared_problems, args)
            format_results.append(row)
            log_wandb_prefixed_row(run, row, model_index, "format")

        format_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "split": args.split,
            "mode": "format_only",
            "sample_count": len(shared_problems),
            "seed": args.seed,
            "models": args.models,
            "results": format_results,
        }
        format_json_path = output_dir / "format_only_summary.json"
        format_csv_path = output_dir / "format_only_summary.csv"
        write_json(format_json_path, format_payload)
        write_csv_generic(format_csv_path, format_results)
        if not args.skip_plot:
            try:
                format_plot_path = plot_format_results(format_results, output_dir)
            except ModuleNotFoundError as exc:
                print(f"Format plot skipped: missing dependency ({exc}).")

    if not args.skip_env:
        print(f"Environment evaluation: {len(shared_problems)} sampled problems")
        for model_index, model_id in enumerate(args.models):
            row = evaluate_model(model_id, shared_problems, args)
            env_results.append(row)
            log_wandb_prefixed_row(run, row, model_index, "env")

        env_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "split": args.split,
            "mode": "environment_rollout",
            "sample_count": len(shared_problems),
            "seed": args.seed,
            "models": args.models,
            "results": env_results,
        }
        env_json_path = output_dir / "env_interaction_summary.json"
        env_csv_path = output_dir / "env_interaction_summary.csv"
        write_json(env_json_path, env_payload)
        write_csv(env_csv_path, env_results)
        if not args.skip_plot:
            try:
                env_plot_path = plot_results(env_results, output_dir)
            except ModuleNotFoundError as exc:
                print(f"Environment plot skipped: missing dependency ({exc}).")

    print("\n--- Format-only Summary ---")
    for row in format_results:
        print(
            f"Model: {row['model_id']} | "
            f"StrictXML: {row['strict_xml_rate'] * 100:.2f}% | "
            f"OverallCompliance: {row['overall_compliance_rate'] * 100:.2f}%"
        )

    print("\n--- Environment Summary ---")
    for row in env_results:
        print(
            f"Model: {row['model_id']} | "
            f"EnvAcceptedTurns: {row['env_format_valid_turn_rate'] * 100:.2f}% | "
            f"RetryUnparsable: {row['retry_unparsable_output_rate'] * 100:.2f}% | "
            f"Success: {row['success_rate'] * 100:.2f}%"
        )

    if format_json_path:
        print(f"\nSaved format JSON: {format_json_path}")
        print(f"Saved format CSV : {format_csv_path}")
        if format_plot_path:
            print(f"Saved format plot: {format_plot_path}")
    if env_json_path:
        print(f"\nSaved env JSON: {env_json_path}")
        print(f"Saved env CSV : {env_csv_path}")
        if env_plot_path:
            print(f"Saved env plot: {env_plot_path}")

    if run is not None:
        if format_json_path:
            run.summary["format_summary_json_path"] = str(format_json_path)
            run.summary["format_summary_csv_path"] = str(format_csv_path)
        if env_json_path:
            run.summary["env_summary_json_path"] = str(env_json_path)
            run.summary["env_summary_csv_path"] = str(env_csv_path)
        if format_plot_path:
            import wandb

            run.log({"format_compliance_plot": wandb.Image(str(format_plot_path))})
            wandb.save(str(format_plot_path), base_path=str(format_plot_path.parent))
        if env_plot_path:
            import wandb

            run.log({"env_interaction_plot": wandb.Image(str(env_plot_path))})
            wandb.save(str(env_plot_path), base_path=str(env_plot_path.parent))
        if format_json_path:
            import wandb

            wandb.save(str(format_json_path), base_path=str(format_json_path.parent))
            wandb.save(str(format_csv_path), base_path=str(format_csv_path.parent))
        if env_json_path:
            import wandb

            wandb.save(str(env_json_path), base_path=str(env_json_path.parent))
            wandb.save(str(env_csv_path), base_path=str(env_csv_path.parent))
        run.finish()


if __name__ == "__main__":
    main()
