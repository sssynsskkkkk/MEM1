from rollout.llm_agent.maze_utils import replay_solution


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    del method, format_score

    result = replay_solution(
        solution_str=solution_str,
        ground_truth=ground_truth,
    )
    return score if result.success else 0.0
