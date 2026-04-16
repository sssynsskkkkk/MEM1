from __future__ import annotations

from collections import deque
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .actions import DIRECTION_DELTAS, DIRECTION_ORDER, move_coordinate


Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


def _canonical_edge(a: Coord, b: Coord) -> Edge:
    return (a, b) if a <= b else (b, a)


def iter_cells(height: int, width: int) -> Iterable[Coord]:
    for row in range(height):
        for col in range(width):
            yield row, col


def in_bounds(coord: Coord, height: int, width: int) -> bool:
    row, col = coord
    return 0 <= row < height and 0 <= col < width


def adjacent_cells(coord: Coord, height: int, width: int) -> List[Tuple[str, Coord]]:
    neighbors: List[Tuple[str, Coord]] = []
    for direction in DIRECTION_ORDER:
        nxt = move_coordinate(coord, direction)
        if in_bounds(nxt, height, width):
            neighbors.append((direction, nxt))
    return neighbors


def build_adjacency(height: int, width: int, open_edges: Sequence[Edge]) -> Dict[Coord, Set[Coord]]:
    adjacency = {cell: set() for cell in iter_cells(height, width)}
    for left, right in open_edges:
        if not in_bounds(left, height, width) or not in_bounds(right, height, width):
            raise ValueError(f"edge contains out-of-bounds cell: {left}, {right}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def serialize_open_edges(open_edges: Sequence[Edge]) -> List[List[List[int]]]:
    rows: List[List[List[int]]] = []
    for left, right in sorted(_canonical_edge(a, b) for a, b in open_edges):
        rows.append([[left[0], left[1]], [right[0], right[1]]])
    return rows


def deserialize_open_edges(serialized_edges: Sequence[Sequence[Sequence[int]]]) -> List[Edge]:
    edges: List[Edge] = []
    for left, right in serialized_edges:
        edge = _canonical_edge((int(left[0]), int(left[1])), (int(right[0]), int(right[1])))
        edges.append(edge)
    return edges


def recursive_backtracking_maze(height: int, width: int, rng: Random) -> List[Edge]:
    if height <= 0 or width <= 0:
        raise ValueError(f"maze dimensions must be positive, got height={height}, width={width}")

    visited: Set[Coord] = set()
    open_edges: Set[Edge] = set()
    start = (0, 0)
    stack = [start]
    visited.add(start)

    while stack:
        current = stack[-1]
        unvisited = [
            neighbor
            for _, neighbor in adjacent_cells(current, height, width)
            if neighbor not in visited
        ]
        if not unvisited:
            stack.pop()
            continue

        nxt = rng.choice(unvisited)
        open_edges.add(_canonical_edge(current, nxt))
        visited.add(nxt)
        stack.append(nxt)

    return sorted(open_edges)


def inject_loops(height: int, width: int, open_edges: Sequence[Edge], rng: Random, loop_injection_ratio: float) -> List[Edge]:
    if loop_injection_ratio <= 0:
        return list(open_edges)

    open_edge_set = {_canonical_edge(a, b) for a, b in open_edges}
    candidates: List[Edge] = []
    for cell in iter_cells(height, width):
        for _, neighbor in adjacent_cells(cell, height, width):
            edge = _canonical_edge(cell, neighbor)
            if edge not in open_edge_set:
                candidates.append(edge)

    deduped_candidates = sorted(set(candidates))
    rng.shuffle(deduped_candidates)
    loop_count = min(len(deduped_candidates), int(round(len(deduped_candidates) * loop_injection_ratio)))

    for edge in deduped_candidates[:loop_count]:
        open_edge_set.add(edge)
    return sorted(open_edge_set)


def bfs_shortest_path(start: Coord, goal: Coord, height: int, width: int, open_edges: Sequence[Edge]) -> List[Coord]:
    adjacency = build_adjacency(height, width, open_edges)
    queue = deque([start])
    parents: Dict[Coord, Optional[Coord]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in sorted(adjacency[current]):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)

    if goal not in parents:
        return []

    path = [goal]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def bfs_shortest_path_length(start: Coord, goal: Coord, height: int, width: int, open_edges: Sequence[Edge]) -> Optional[int]:
    path = bfs_shortest_path(start, goal, height, width, open_edges)
    if not path:
        return None
    return max(len(path) - 1, 0)


def path_to_actions(path: Sequence[Coord]) -> List[str]:
    actions: List[str] = []
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        for direction, direction_delta in DIRECTION_DELTAS.items():
            if delta == direction_delta:
                actions.append(direction)
                break
        else:
            raise ValueError(f"non-adjacent path step: {current} -> {nxt}")
    return actions


def choose_start_goal(height: int, width: int, open_edges: Sequence[Edge], min_shortest_path: int, rng: Random) -> Tuple[Coord, Coord, List[Coord]]:
    candidate_paths: List[List[Coord]] = []
    for start in iter_cells(height, width):
        for goal in iter_cells(height, width):
            if start >= goal:
                continue
            path = bfs_shortest_path(start, goal, height, width, open_edges)
            if not path:
                continue
            distance = len(path) - 1
            if distance >= min_shortest_path:
                candidate_paths.append(path)

    if not candidate_paths:
        raise ValueError(f"could not find start/goal with shortest path >= {min_shortest_path}")

    candidate_paths.sort(key=len, reverse=True)
    longest_length = len(candidate_paths[0])
    longest_paths = [path for path in candidate_paths if len(path) == longest_length]
    chosen_path = rng.choice(longest_paths)
    return chosen_path[0], chosen_path[-1], chosen_path


def generate_maze_spec(
    *,
    width: int,
    height: int,
    min_shortest_path: int,
    loop_injection_ratio: float = 0.0,
    max_steps: Optional[int] = None,
    max_steps_factor: float = 2.0,
    observation_mode: str = "move_plus_local4",
    reveal_position: bool = False,
    step_penalty: float = -0.01,
    blocked_move_penalty: float = -0.02,
    goal_reward: float = 1.0,
    first_visit_bonus: float = 0.01,
    seed: Optional[int] = None,
    max_generation_attempts: int = 128,
) -> Dict:
    rng = Random(seed)

    for attempt_idx in range(max_generation_attempts):
        trial_seed = rng.randint(0, 10**9) if seed is None else seed + attempt_idx
        trial_rng = Random(trial_seed)
        open_edges = recursive_backtracking_maze(height, width, trial_rng)
        open_edges = inject_loops(height, width, open_edges, trial_rng, loop_injection_ratio)

        try:
            start, goal, optimal_path = choose_start_goal(height, width, open_edges, min_shortest_path, trial_rng)
        except ValueError:
            continue

        optimal_steps = max(len(optimal_path) - 1, 1)
        final_max_steps = max_steps
        if final_max_steps is None:
            final_max_steps = max(int(round(optimal_steps * max_steps_factor)), optimal_steps + 1)

        return {
            "width": int(width),
            "height": int(height),
            "start": [start[0], start[1]],
            "goal": [goal[0], goal[1]],
            "open_edges": serialize_open_edges(open_edges),
            "optimal_path": [[row, col] for row, col in optimal_path],
            "optimal_steps": int(optimal_steps),
            "max_steps": int(final_max_steps),
            "observation_mode": observation_mode,
            "reveal_position": bool(reveal_position),
            "step_penalty": float(step_penalty),
            "blocked_move_penalty": float(blocked_move_penalty),
            "goal_reward": float(goal_reward),
            "first_visit_bonus": float(first_visit_bonus),
            "loop_injection_ratio": float(loop_injection_ratio),
            "maze_seed": int(trial_seed),
        }

    raise RuntimeError(
        "failed to generate a solvable maze with the requested minimum shortest-path distance "
        f"after {max_generation_attempts} attempts"
    )
