from __future__ import annotations

import re
from typing import Dict, Tuple


Coord = Tuple[int, int]

DIRECTION_ORDER = ("UP", "DOWN", "LEFT", "RIGHT")
DIRECTION_DELTAS: Dict[str, Coord] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

STRICT_ACTION_RE = re.compile(r"^\s*(UP|DOWN|LEFT|RIGHT)\s*$", re.IGNORECASE)
RELAXED_ACTION_RE = re.compile(r"\b(UP|DOWN|LEFT|RIGHT)\b", re.IGNORECASE)


def parse_direction_action(action_text: str) -> str:
    text = action_text.strip()
    match = STRICT_ACTION_RE.match(text)
    if not match:
        match = RELAXED_ACTION_RE.search(text)
    if not match:
        raise ValueError(f"invalid maze action: {action_text}")
    return match.group(1).upper()


def move_coordinate(coord: Coord, direction: str) -> Coord:
    delta_row, delta_col = DIRECTION_DELTAS[direction]
    return coord[0] + delta_row, coord[1] + delta_col
