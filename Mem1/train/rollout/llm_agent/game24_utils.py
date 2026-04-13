import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence


STRICT_OUTPUT_RE = re.compile(
    r"^\s*"
    r"(?P<reasoning><reasoning>.*?</reasoning>)\s*"
    r"(?P<summary><summary>.*?</summary>)\s*"
    r"(?P<action><action>.*?</action>)\s*$",
    re.DOTALL,
)

NO_REASONING_OUTPUT_RE = re.compile(
    r"^\s*"
    r"(?:(?P<reasoning><reasoning>.*?</reasoning>)\s*)?"
    r"(?P<summary><summary>.*?</summary>)\s*"
    r"(?P<action><action>.*?</action>)\s*$",
    re.DOTALL,
)

TAG_CONTENT_RE = {
    "reasoning": re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL),
    "summary": re.compile(r"<summary>(.*?)</summary>", re.DOTALL),
    "action": re.compile(r"<action>(.*?)</action>", re.DOTALL),
}

STRICT_ACTION_RE = re.compile(r"^\s*([^\s]+)\s*([+\-*/])\s*([^\s]+)\s*$")
RELAXED_ACTION_RE = re.compile(r"([+-]?\d+(?:/\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:/\d+)?)")


@dataclass
class ParsedOutput:
    raw_text: str
    format_ok: bool
    reasoning_text: str
    summary_text: str
    action_text: str
    reasoning_block: str
    summary_block: str
    action_block: str


@dataclass
class ActionOutcome:
    is_valid: bool
    next_state: List[Fraction]
    result: Optional[Fraction] = None
    error: str = ""


@dataclass
class RolloutResult:
    success: bool
    invalid_action: bool
    steps: int
    final_state: List[Fraction]
    reason: str


@dataclass
class RetryReplayResult:
    success: bool
    invalid_action_count: int
    total_steps: int
    retries_used: int


def _strip_assistant_prefix(text: str) -> str:
    if "Assistant:" in text:
        return text.split("Assistant:", 1)[1]
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant", 1)[1]
    return text


def truncate_after_action(text: str) -> str:
    base = _strip_assistant_prefix(text)
    if "</action>" in base:
        return base.split("</action>", 1)[0] + "</action>"
    return base.strip()


def extract_tag_text(tag: str, text: str) -> str:
    match = TAG_CONTENT_RE[tag].search(text)
    if not match:
        return ""
    return match.group(1).strip()


def extract_tag_block(tag: str, text: str) -> str:
    inner = extract_tag_text(tag, text)
    if not inner:
        return ""
    return f"<{tag}>{inner}</{tag}>"


def parse_model_output(text: str, require_reasoning: bool = True) -> ParsedOutput:
    trimmed = truncate_after_action(text)
    pattern = STRICT_OUTPUT_RE if require_reasoning else NO_REASONING_OUTPUT_RE
    match = pattern.match(trimmed)

    if match:
        reasoning_block = match.groupdict().get("reasoning", "") or ""
        summary_block = match.group("summary")
        action_block = match.group("action")
        return ParsedOutput(
            raw_text=trimmed,
            format_ok=True,
            reasoning_text=extract_tag_text("reasoning", reasoning_block) if reasoning_block else "",
            summary_text=extract_tag_text("summary", summary_block),
            action_text=extract_tag_text("action", action_block),
            reasoning_block=reasoning_block,
            summary_block=summary_block,
            action_block=action_block,
        )

    return ParsedOutput(
        raw_text=trimmed,
        format_ok=False,
        reasoning_text=extract_tag_text("reasoning", trimmed),
        summary_text=extract_tag_text("summary", trimmed),
        action_text=extract_tag_text("action", trimmed),
        reasoning_block=extract_tag_block("reasoning", trimmed),
        summary_block=extract_tag_block("summary", trimmed),
        action_block=extract_tag_block("action", trimmed),
    )


def parse_fraction(text: str) -> Fraction:
    value = text.strip()
    if not value:
        raise ValueError("empty numeric token")
    return Fraction(value)


def parse_action(action_text: str):
    match = STRICT_ACTION_RE.match(action_text.strip())
    if not match:
        match = RELAXED_ACTION_RE.search(action_text)
    if not match:
        raise ValueError(f"invalid action format: {action_text}")
    left, op, right = match.groups()
    return parse_fraction(left), op, parse_fraction(right)


def _sort_key(value: Fraction):
    return (float(value), value.numerator, value.denominator)


def canonicalize_state(values: Sequence[Fraction]) -> List[Fraction]:
    return sorted(list(values), key=_sort_key)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_state(values: Sequence[Fraction]) -> str:
    return ", ".join(format_fraction(v) for v in canonicalize_state(values))


def extract_puzzle_spec(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    numbers = ground_truth.get("numbers")
    if numbers is None:
        raise ValueError("ground_truth must contain 'numbers'")
    target = ground_truth.get("target", 24)
    max_steps = int(ground_truth.get("max_steps", max(len(numbers) - 1, 1)))
    return {
        "numbers": [Fraction(str(x)) for x in numbers],
        "target": Fraction(str(target)),
        "max_steps": max_steps,
    }


def _match_operand(target: Fraction, state: Sequence[Fraction], used_indices: Sequence[int], tolerance: float) -> Optional[int]:
    best_index = None
    best_error = None
    used = set(used_indices)
    for idx, current in enumerate(state):
        if idx in used:
            continue
        error = abs(float(current - target))
        if error <= tolerance and (best_error is None or error < best_error):
            best_index = idx
            best_error = error
    return best_index


def apply_action_to_state(state: Sequence[Fraction], action_text: str, tolerance: float = 1e-5) -> ActionOutcome:
    try:
        left, op, right = parse_action(action_text)
    except Exception as exc:
        return ActionOutcome(is_valid=False, next_state=list(state), error=str(exc))

    left_idx = _match_operand(left, state, used_indices=[], tolerance=tolerance)
    if left_idx is None:
        return ActionOutcome(is_valid=False, next_state=list(state), error="left operand not in state")

    right_idx = _match_operand(right, state, used_indices=[left_idx], tolerance=tolerance)
    if right_idx is None:
        return ActionOutcome(is_valid=False, next_state=list(state), error="right operand not in state")

    a = state[left_idx]
    b = state[right_idx]

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            return ActionOutcome(is_valid=False, next_state=list(state), error="division by zero")
        result = a / b
    else:
        return ActionOutcome(is_valid=False, next_state=list(state), error=f"unsupported operator: {op}")

    next_state = [value for idx, value in enumerate(state) if idx not in {left_idx, right_idx}]
    next_state.append(result)
    return ActionOutcome(
        is_valid=True,
        next_state=canonicalize_state(next_state),
        result=result,
    )


def extract_action_texts(solution_str: str) -> List[str]:
    base = _strip_assistant_prefix(solution_str)
    return [match.group(1).strip() for match in TAG_CONTENT_RE["action"].finditer(base)]


def replay_actions(
    numbers: Sequence[Any],
    actions: Sequence[str],
    target: Any = 24,
    max_steps: Optional[int] = None,
    tolerance: float = 1e-5,
) -> RolloutResult:
    state = canonicalize_state(Fraction(str(x)) for x in numbers)
    target_value = Fraction(str(target))
    if max_steps is None:
        max_steps = max(len(state) - 1, 1)

    steps = 0
    for action in actions[:max_steps]:
        outcome = apply_action_to_state(state, action, tolerance=tolerance)
        if not outcome.is_valid:
            return RolloutResult(
                success=False,
                invalid_action=True,
                steps=steps + 1,
                final_state=state,
                reason=outcome.error,
            )
        state = outcome.next_state
        steps += 1
        if len(state) == 1:
            break

    success = len(state) == 1 and abs(float(state[0] - target_value)) <= tolerance
    reason = "success" if success else "terminal_non_success"
    return RolloutResult(
        success=success,
        invalid_action=False,
        steps=steps,
        final_state=state,
        reason=reason,
    )


def replay_with_retries(
    ground_truth: Dict[str, Any],
    actions: Sequence[str],
    max_total_turns: Optional[int] = None,
    tolerance: float = 1e-5,
) -> RetryReplayResult:
    spec = extract_puzzle_spec(ground_truth)
    initial_state = canonicalize_state(spec["numbers"])
    state = list(initial_state)
    target = spec["target"]
    max_steps = spec["max_steps"]

    invalid_action_count = 0
    total_steps = 0
    retries_used = 0
    steps_in_attempt = 0

    for action in actions:
        if max_total_turns is not None and total_steps >= max_total_turns:
            break

        outcome = apply_action_to_state(state, action, tolerance=tolerance)
        total_steps += 1

        if not outcome.is_valid:
            invalid_action_count += 1
            retries_used += 1
            state = list(initial_state)
            steps_in_attempt = 0
            continue

        state = outcome.next_state
        steps_in_attempt += 1

        if len(state) == 1:
            if abs(float(state[0] - target)) <= tolerance:
                return RetryReplayResult(
                    success=True,
                    invalid_action_count=invalid_action_count,
                    total_steps=total_steps,
                    retries_used=retries_used,
                )
            retries_used += 1
            state = list(initial_state)
            steps_in_attempt = 0
            continue

        if steps_in_attempt >= max_steps:
            retries_used += 1
            state = list(initial_state)
            steps_in_attempt = 0

    return RetryReplayResult(
        success=False,
        invalid_action_count=invalid_action_count,
        total_steps=total_steps,
        retries_used=retries_used,
    )
