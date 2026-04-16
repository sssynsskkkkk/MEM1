from __future__ import annotations

from typing import Dict, Optional, Tuple

from .actions import DIRECTION_ORDER


Coord = Tuple[int, int]


def format_coordinate(coord: Coord) -> str:
    return f"({coord[0]}, {coord[1]})"


def format_local_view(local_view: Dict[str, bool]) -> str:
    parts = []
    for direction in DIRECTION_ORDER:
        cell_state = "OPEN" if local_view.get(direction, False) else "WALL"
        parts.append(f"{direction}={cell_state}")
    return ", ".join(parts)


def format_observation(
    *,
    turn_index: int,
    turns_left: int,
    previous_result: str,
    goal_reached: bool,
    observation_mode: str,
    last_action: Optional[str] = None,
    local_view: Optional[Dict[str, bool]] = None,
    reveal_position: bool = False,
    position: Optional[Coord] = None,
) -> str:
    lines = [
        "<observation>",
        f"turn_index: {turn_index}",
        f"turns_left: {turns_left}",
        f"previous_result: {previous_result}",
    ]
    if last_action:
        lines.append(f"last_action: {last_action}")
    lines.append(f"goal_status: {'reached' if goal_reached else 'not_reached'}")
    if observation_mode == "move_plus_local4" and local_view is not None:
        lines.append(f"local_view: {format_local_view(local_view)}")
    if reveal_position and position is not None:
        lines.append(f"current_coordinate: {format_coordinate(position)}")
    lines.append("</observation>")
    return "\n".join(lines)
