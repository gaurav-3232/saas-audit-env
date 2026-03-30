"""Deterministic graders for SaaSAuditEnv.

Each grader takes the final episode state and returns a score in [0.0, 1.0].
Scoring is fully deterministic — given the same end state, it always returns
the same score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.env import SaaSAuditEnv


def grade_episode(env: "SaaSAuditEnv") -> float:
    """Dispatch to the appropriate task grader."""
    graders = {
        "task_easy": _grade_easy,
        "task_medium": _grade_medium,
        "task_hard": _grade_hard,
        "task_startup": _grade_startup,
        "task_merger": _grade_merger,
    }
    grader = graders.get(env.current_task.task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task: {env.current_task.task_id}")
    return round(grader(env), 4)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _savings_score(achieved: float, target: float) -> float:
    """Fractional score based on savings achieved vs target."""
    if target <= 0:
        return 1.0
    return min(1.0, achieved / target)


def _critical_tool_penalty(env: "SaaSAuditEnv") -> float:
    """Returns 1.0 if all critical tools are intact, else 0.0."""
    for tool_id in env.current_task.must_preserve:
        if tool_id in env.cancelled_tools:
            return 0.0
    return 1.0


def _invalid_action_penalty(env: "SaaSAuditEnv") -> float:
    """Returns a multiplier in [0.5, 1.0] penalising excessive invalid actions."""
    n = env.invalid_action_count
    # Up to 3 invalid actions: no penalty; each additional reduces by 0.05
    penalty_steps = max(0, n - 3)
    return max(0.5, 1.0 - penalty_steps * 0.05)


def _submission_bonus(env: "SaaSAuditEnv") -> float:
    """Returns 0.1 bonus if a recommendation was submitted, else 0.0."""
    return 0.1 if env.recommendation_submitted else 0.0


# ---------------------------------------------------------------------------
# Per-task graders
# ---------------------------------------------------------------------------

def _grade_easy(env: "SaaSAuditEnv") -> float:
    """
    Grade for task_easy.

    Score breakdown:
        50% — savings achieved relative to $400 target
        20% — critical tools preserved (binary)
        20% — at least one seat-reduction was applied to a wasteful tool
        10% — submission bonus
    """
    savings_frac = _savings_score(env.total_savings_achieved, env.current_task.target_savings)
    critical_ok = _critical_tool_penalty(env)

    # Did the agent actually reduce seats on at least one tool?
    seat_reductions = sum(
        1 for r in env.action_history if r.action_type == "reduce_seats" and r.success
    )
    seat_action_score = 1.0 if seat_reductions >= 1 else 0.0

    submission = _submission_bonus(env)

    raw = (
        0.50 * savings_frac
        + 0.20 * critical_ok
        + 0.20 * seat_action_score
        + 0.10 * submission
    )
    return min(1.0, raw * _invalid_action_penalty(env))


def _grade_medium(env: "SaaSAuditEnv") -> float:
    """
    Grade for task_medium.

    Score breakdown:
        40% — savings achieved relative to $900 target
        25% — critical tools preserved
        20% — at least one duplicate/overlap handled (cancel or merge)
        10% — seat right-sizing on wasteful tools
         5% — submission bonus
    """
    savings_frac = _savings_score(env.total_savings_achieved, env.current_task.target_savings)
    critical_ok = _critical_tool_penalty(env)

    # Overlap handling: cancel or merge of any tool in an overlap group
    safe_cancels = set(env.current_task.safe_to_cancel)
    overlap_handled = any(
        (r.action_type in ("cancel_subscription", "merge_tools") and r.success)
        for r in env.action_history
        if r.details.get("tool_id") in safe_cancels
        or r.details.get("cancel_tool_id") in safe_cancels
    )
    overlap_score = 1.0 if overlap_handled else 0.0

    # Seat right-sizing
    seat_reductions = sum(
        1 for r in env.action_history if r.action_type == "reduce_seats" and r.success
    )
    seat_score = min(1.0, seat_reductions / 2)  # full credit at 2+ reductions

    submission = _submission_bonus(env)

    raw = (
        0.40 * savings_frac
        + 0.25 * critical_ok
        + 0.20 * overlap_score
        + 0.10 * seat_score
        + 0.05 * submission
    )
    return min(1.0, raw * _invalid_action_penalty(env))


def _grade_hard(env: "SaaSAuditEnv") -> float:
    """
    Grade for task_hard.

    Score breakdown:
        40% — savings achieved relative to $2,500 target
        25% — critical tools preserved (binary)
        15% — correctly handled 2+ overlap groups
        10% — seat right-sizing (3+ successful reductions for full credit)
        10% — submission with accurate savings estimate
    """
    savings_frac = _savings_score(env.total_savings_achieved, env.current_task.target_savings)
    critical_ok = _critical_tool_penalty(env)

    # Overlap groups handled: any cancel/merge in each group
    groups_handled = 0
    for group_tools in env.current_task.overlap_groups.values():
        group_set = set(group_tools)
        handled = any(
            (r.action_type in ("cancel_subscription", "merge_tools") and r.success)
            and (
                r.details.get("tool_id") in group_set
                or r.details.get("cancel_tool_id") in group_set
            )
            for r in env.action_history
        )
        if handled:
            groups_handled += 1
    total_groups = max(1, len(env.current_task.overlap_groups))
    overlap_score = min(1.0, groups_handled / total_groups)

    # Seat reductions — full credit at 3+
    seat_reductions = sum(
        1 for r in env.action_history if r.action_type == "reduce_seats" and r.success
    )
    seat_score = min(1.0, seat_reductions / 3)

    # Submission accuracy: reported savings within 15% of actual
    submission_score = 0.0
    if env.recommendation_submitted and env.submitted_savings_estimate is not None:
        actual = env.total_savings_achieved
        if actual > 0:
            ratio = env.submitted_savings_estimate / actual
            submission_score = 1.0 if 0.85 <= ratio <= 1.15 else 0.5
        else:
            submission_score = 0.0

    raw = (
        0.40 * savings_frac
        + 0.25 * critical_ok
        + 0.15 * overlap_score
        + 0.10 * seat_score
        + 0.10 * submission_score
    )
    return min(1.0, raw * _invalid_action_penalty(env))


def _grade_startup(env: "SaaSAuditEnv") -> float:
    """
    Grade for task_startup.

    Score breakdown:
        35% — savings achieved relative to $1,200 target
        20% — critical tools preserved (binary)
        20% — plan downgrades performed (full credit at 2+)
        15% — overlap handled (analytics or docs consolidated)
        10% — submission bonus
    """
    savings_frac = _savings_score(
        env.total_savings_achieved, env.current_task.target_savings
    )
    critical_ok = _critical_tool_penalty(env)

    # Plan downgrades — this task rewards downgrade actions specifically
    downgrades = sum(
        1 for r in env.action_history
        if r.action_type == "downgrade_plan" and r.success
    )
    downgrade_score = min(1.0, downgrades / 2)

    # Overlap handling
    safe_cancels = set(env.current_task.safe_to_cancel)
    overlap_handled = any(
        (r.action_type in ("cancel_subscription", "merge_tools") and r.success)
        for r in env.action_history
        if r.details.get("tool_id") in safe_cancels
        or r.details.get("cancel_tool_id") in safe_cancels
    )
    overlap_score = 1.0 if overlap_handled else 0.0

    submission = _submission_bonus(env)

    raw = (
        0.35 * savings_frac
        + 0.20 * critical_ok
        + 0.20 * downgrade_score
        + 0.15 * overlap_score
        + 0.10 * submission
    )
    return min(1.0, raw * _invalid_action_penalty(env))


def _grade_merger(env: "SaaSAuditEnv") -> float:
    """
    Grade for task_merger.

    Score breakdown:
        35% — savings achieved relative to $4,000 target
        20% — critical tools preserved (binary)
        25% — overlap groups consolidated (7 groups, full credit at 5+)
        10% — seat right-sizing (4+ successful reductions for full credit)
        10% — submission with accurate savings estimate
    """
    savings_frac = _savings_score(
        env.total_savings_achieved, env.current_task.target_savings
    )
    critical_ok = _critical_tool_penalty(env)

    # Overlap groups handled: consolidation across the 7 merger pairs
    groups_handled = 0
    for group_tools in env.current_task.overlap_groups.values():
        group_set = set(group_tools)
        handled = any(
            (r.action_type in ("cancel_subscription", "merge_tools") and r.success)
            and (
                r.details.get("tool_id") in group_set
                or r.details.get("cancel_tool_id") in group_set
            )
            for r in env.action_history
        )
        if handled:
            groups_handled += 1
    total_groups = max(1, len(env.current_task.overlap_groups))
    overlap_score = min(1.0, groups_handled / min(5, total_groups))

    # Seat reductions — full credit at 4+
    seat_reductions = sum(
        1 for r in env.action_history
        if r.action_type == "reduce_seats" and r.success
    )
    seat_score = min(1.0, seat_reductions / 4)

    # Submission accuracy
    submission_score = 0.0
    if env.recommendation_submitted and env.submitted_savings_estimate is not None:
        actual = env.total_savings_achieved
        if actual > 0:
            ratio = env.submitted_savings_estimate / actual
            submission_score = 1.0 if 0.85 <= ratio <= 1.15 else 0.5
        else:
            submission_score = 0.0

    raw = (
        0.35 * savings_frac
        + 0.20 * critical_ok
        + 0.25 * overlap_score
        + 0.10 * seat_score
        + 0.10 * submission_score
    )
    return min(1.0, raw * _invalid_action_penalty(env))
