from rollout.llm_agent.game24_utils import extract_action_texts, replay_with_retries


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    del method, format_score

    actions = extract_action_texts(solution_str)
    result = replay_with_retries(
        ground_truth=ground_truth,
        actions=actions,
        max_total_turns=len(actions),
    )
    return score if result.success else 0.0
